//! 基于 cpal 的采集实现：麦克风（全平台）与桌面回环。

use crate::resample::Resampler;
use crate::source::{AudioSource, Capture, Error, PcmSink, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};

fn names(devices: impl Iterator<Item = Device>) -> Result<Vec<String>> {
    devices
        .map(|d| d.name().map_err(|e| Error::Device(e.to_string())))
        .collect()
}

fn find_named(mut devices: impl Iterator<Item = Device>, name: &str) -> Option<Device> {
    devices.find(|d| d.name().is_ok_and(|n| n == name))
}

/// 设备原生格式 → 16kHz 单声道 i16 的转换状态（音频回调内逐块使用）
struct To16k {
    res: Resampler,
    ch: usize,
    mono: Vec<f32>,
    out: Vec<i16>,
    sink: PcmSink,
}

impl To16k {
    fn new(in_rate: u32, ch: u16, sink: PcmSink) -> Self {
        Self {
            res: Resampler::new(in_rate, 16000),
            ch: ch as usize,
            mono: Vec::new(),
            out: Vec::new(),
            sink,
        }
    }

    /// 输入任意通道数的交错 f32 样本（-1..1），下混单声道后重采样投递
    fn feed(&mut self, samples: impl Iterator<Item = f32>) {
        self.mono.clear();
        self.out.clear();
        let (mut acc, mut n) = (0.0f32, 0usize);
        for s in samples {
            acc += s;
            n += 1;
            if n == self.ch {
                self.mono.push(acc / self.ch as f32);
                (acc, n) = (0.0, 0);
            }
        }
        // 不满一帧的尾部样本丢弃（正常回调不会出现）
        self.res.push(&self.mono, &mut self.out);
        (self.sink)(&self.out);
    }
}

/// 在设备上建立输入流并播放（采集即开始）
fn spawn_input(
    dev: &Device,
    cfg: &StreamConfig,
    fmt: SampleFormat,
    sink: PcmSink,
) -> Result<Stream> {
    let (rate, ch) = (cfg.sample_rate.0, cfg.channels);
    let err = |e: cpal::StreamError| eprintln!("[wxasr-provider] 音频流错误: {e}");
    let stream = match fmt {
        SampleFormat::F32 => {
            let mut c = To16k::new(rate, ch, sink);
            dev.build_input_stream(
                cfg,
                move |d: &[f32], _| c.feed(d.iter().copied()),
                err,
                None,
            )
        }
        SampleFormat::I16 => {
            let mut c = To16k::new(rate, ch, sink);
            dev.build_input_stream(
                cfg,
                move |d: &[i16], _| c.feed(d.iter().map(|&v| v as f32 / 32768.0)),
                err,
                None,
            )
        }
        SampleFormat::U16 => {
            let mut c = To16k::new(rate, ch, sink);
            dev.build_input_stream(
                cfg,
                move |d: &[u16], _| c.feed(d.iter().map(|&v| v as f32 / 32768.0 - 1.0)),
                err,
                None,
            )
        }
        other => return Err(Error::Unsupported(format!("样本格式 {other:?}"))),
    }
    .map_err(|e| Error::Device(format!("建立输入流失败: {e}")))?;
    stream
        .play()
        .map_err(|e| Error::Device(format!("启动音频流失败: {e}")))?;
    Ok(stream)
}

/// 麦克风采集（全平台）
pub struct Microphone;

impl Default for Microphone {
    fn default() -> Self {
        Self::new()
    }
}

impl Microphone {
    pub fn new() -> Self {
        Self
    }
}

impl AudioSource for Microphone {
    fn name(&self) -> &'static str {
        "microphone"
    }

    fn devices(&self) -> Result<Vec<String>> {
        let devices = cpal::default_host()
            .input_devices()
            .map_err(|e| Error::Device(e.to_string()))?;
        names(devices)
    }

    fn start(&self, device: Option<&str>, sink: PcmSink) -> Result<Capture> {
        let host = cpal::default_host();
        let dev = match device {
            Some(name) => {
                let devices = host
                    .input_devices()
                    .map_err(|e| Error::Device(e.to_string()))?;
                find_named(devices, name)
                    .ok_or_else(|| Error::Device(format!("找不到麦克风: {name}")))?
            }
            None => host
                .default_input_device()
                .ok_or_else(|| Error::Device("没有可用的输入设备".to_string()))?,
        };
        let cfg = dev
            .default_input_config()
            .map_err(|e| Error::Device(format!("读取输入配置失败: {e}")))?;
        let fmt = cfg.sample_format();
        let stream = spawn_input(&dev, &cfg.into(), fmt, sink)?;
        Ok(Capture::new(stream))
    }
}

/// 桌面回环采集（捕获系统正在播放的声音）：
/// - Windows：WASAPI loopback，直接在输出设备上建输入流
/// - Linux：PulseAudio/PipeWire 的 monitor 源（以输入设备形式出现）
/// - macOS：需安装虚拟设备（如 BlackHole），它同样以输入设备形式出现
pub struct DesktopAudio;

impl Default for DesktopAudio {
    fn default() -> Self {
        Self::new()
    }
}

impl DesktopAudio {
    pub fn new() -> Self {
        Self
    }

    #[cfg(target_os = "windows")]
    fn start_impl(&self, device: Option<&str>, sink: PcmSink) -> Result<Capture> {
        let host = cpal::default_host();
        let dev = match device {
            Some(name) => {
                let devices = host
                    .output_devices()
                    .map_err(|e| Error::Device(e.to_string()))?;
                find_named(devices, name)
                    .ok_or_else(|| Error::Device(format!("找不到输出设备: {name}")))?
            }
            None => host
                .default_output_device()
                .ok_or_else(|| Error::Device("没有可用的输出设备".to_string()))?,
        };
        // loopback：以输出设备的混音格式在其上建立输入流
        let cfg = dev
            .default_output_config()
            .map_err(|e| Error::Device(format!("读取输出配置失败: {e}")))?;
        let fmt = cfg.sample_format();
        let stream = spawn_input(&dev, &cfg.into(), fmt, sink)?;
        Ok(Capture::new(stream))
    }

    #[cfg(not(target_os = "windows"))]
    fn start_impl(&self, device: Option<&str>, sink: PcmSink) -> Result<Capture> {
        let host = cpal::default_host();
        let inputs: Vec<Device> = host
            .input_devices()
            .map_err(|e| Error::Device(e.to_string()))?
            .collect();
        let dev = match device {
            Some(name) => find_named(inputs.into_iter(), name)
                .ok_or_else(|| Error::Device(format!("找不到设备: {name}")))?,
            None => inputs
                .iter()
                .find(|d| {
                    d.name()
                        .map(|n| n.to_lowercase().contains("monitor"))
                        .unwrap_or(false)
                })
                .cloned()
                .or_else(|| host.default_input_device())
                .ok_or_else(|| {
                    Error::Unsupported(
                        "未找到桌面回环设备：Linux 需要 PulseAudio/PipeWire monitor 源，\
                         macOS 需要安装 BlackHole 等虚拟设备",
                    )
                })?,
        };
        let cfg = dev
            .default_input_config()
            .map_err(|e| Error::Device(format!("读取输入配置失败: {e}")))?;
        let fmt = cfg.sample_format();
        let stream = spawn_input(&dev, &cfg.into(), fmt, sink)?;
        Ok(Capture::new(stream))
    }
}

impl AudioSource for DesktopAudio {
    fn name(&self) -> &'static str {
        "desktop"
    }

    fn devices(&self) -> Result<Vec<String>> {
        #[cfg(target_os = "windows")]
        {
            let devices = cpal::default_host()
                .output_devices()
                .map_err(|e| Error::Device(e.to_string()))?;
            names(devices)
        }
        #[cfg(not(target_os = "windows"))]
        {
            let devices = cpal::default_host()
                .input_devices()
                .map_err(|e| Error::Device(e.to_string()))?;
            names(devices)
        }
    }

    fn start(&self, device: Option<&str>, sink: PcmSink) -> Result<Capture> {
        self.start_impl(device, sink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn to16k_downmixes_and_resamples() {
        // 48k 立体声 [L=1, R=-1] ×4800 帧 → 单声道抵消为 0，
        // 数量 4800/3 = 1600 个 i16，全部应为 0
        let logged = Arc::new(Mutex::new(Vec::new()));
        let sink: PcmSink = {
            let logged = logged.clone();
            Arc::new(move |pcm: &[i16]| logged.lock().unwrap().extend_from_slice(pcm))
        };
        let mut c = To16k::new(48000, 2, sink);
        let stereo: Vec<f32> = (0..4800).flat_map(|_| [1.0f32, -1.0f32]).collect();
        c.feed(stereo.into_iter());
        let out = logged.lock().unwrap();
        assert!((out.len() as i64 - 1600).abs() <= 2, "len={}", out.len());
        assert!(out.iter().all(|&v| v == 0), "左右抵消后应为静音");
    }
}
