//! wxasr 后端:统一层的第一个 [`AsrBackend`](crate::AsrBackend) 实现。
//!
//! 会话语义全部在本模块的 [`WxasrStream`] 里(自旧版专用 worker 原样
//! 搬来,逐行等价):内置轻量 VAD 提供静音信号,
//! [`RotatePolicy`](crate::policy::RotatePolicy) 决定上下文轮换,中间结果按
//! 计算成本节流。
//!
//! # 上下文轮换
//!
//! [`wxasr::Stream`] 的内部状态(特征帧缓冲约 16KB/s、已定稿 token)
//! 随会话时长线性增长,长会话会持续挤压内存并拖慢中间结果。会话按
//! [`RotatePolicy`](crate::policy::RotatePolicy) 在合适的时机定稿当前流、发出
//! [`Commit`](crate::AsrEvent::Commit) 并接续(新流携带切点前 1s 左文与切点后
//! 未定稿尾部,切点取在字间停顿处——说话中硬切也不丢字);会话结束时
//! 最后一段以 [`End`](crate::AsrEvent::End) 定稿。会话不保留
//! 已定稿文本,资源占用与会话总长无关。
//!
//! 默认策略为 [`Punct`](crate::policy::RotatePolicy::Punct)
//! `{ soft_ms: 10s, hard_ms: 20s }`:累计 10s 后遇定稿标点即轮换,
//! `Commit` 以标点收尾;累计 20s 即使无标点(音乐/噪声/连续说话)也强制
//! 轮换。[`Auto`](crate::policy::RotatePolicy::Auto) 改为静音触发:软上限后于
//! 下一次 ≥500ms 静音处轮换。注意硬上限可能在说话中触发,定稿解码
//! (约半秒)期间会话暂停消费音频(通道不丢数据,只是中间结果晚到)。
//!
//! # 静音检测
//!
//! 内置轻量自适应 VAD:判定用 25ms 子块的 RMS 峰值,噪声底取静音段
//! RMS 的滑动均值(初值为首个块的 RMS)。数字静音与不同增益的麦克风
//! 均可工作;注意初值取自首个块,若开口极快请先喂一小段环境音。
//! VAD 不直接面向用户,它为轮换策略提供静音信号。

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::policy::{RotateCtx, RotatePolicy};
use crate::preprocess::InputConverter;
use crate::{AsrBackend, AsrEvent, AsrEvents, AsrSession, AsrText, AudioIn, Config};
use wxasr::{Model, Stream};

/// wxasr 后端:每次 [`open`](AsrBackend::open) 从共享的 `Arc<Model>`
/// 创建一个流式会话。权重与内部计算线程池全局复用,每会话仅各自的
/// 流式状态(特征/VAD)
pub struct WxasrBackend {
    model: Arc<Model>,
}

impl WxasrBackend {
    pub fn new(model: Arc<Model>) -> Self {
        WxasrBackend { model }
    }
}

impl AsrBackend for WxasrBackend {
    fn open(&self, cfg: &Config, audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>> {
        // wxasr 只吃 16 kHz 单声道:调用方声明的格式在这里转换。
        // 16 kHz 单声道直通,与历史行为逐字节一致。
        let convert = InputConverter::new(audio)
            .map_err(|reason| eyre::eyre!("wxasr 不支持该音频格式: {reason}"))?;
        Ok(Box::new(WxasrStream {
            stream: Some(self.model.stream()),
            convert,
            vad: Vad::default(),
            quiet_ms: 0,  // 音频尾部连续静音(描述音频,轮换不清零)
            stream_ms: 0, // 当前流累计音频
            last_emit: Instant::now(),
            last_cost: Duration::ZERO,
            last_partial: String::new(),
            partial: cfg.partial,
            partial_min_fresh: cfg.partial_min_fresh,
            rotate: cfg.rotate.clone(),
        }))
    }
}

/// wxasr 流式会话:拥有 [`wxasr::Stream`] 与全部会话语义(VAD、中间
/// 结果节流、上下文轮换),由统一 worker 在专用识别线程里驱动
pub struct WxasrStream {
    /// [`rotate`](Stream::rotate)/[`finish`](Stream::finish) 按值消耗
    /// `Stream`,轮换时经此 `Option` 换出换入;`finish` 后为 `None`
    /// (会话已结束,后续 `push` 空转)
    stream: Option<Stream>,
    /// 调用方格式 → 16 kHz 单声道(直通时零开销)
    convert: InputConverter,
    vad: Vad,
    /// 音频尾部连续静音(描述音频,轮换不清零)
    quiet_ms: u32,
    /// 当前流累计音频
    stream_ms: u32,
    last_emit: Instant,
    last_cost: Duration,
    last_partial: String,
    partial: bool,
    partial_min_fresh: usize,
    rotate: RotatePolicy,
}

impl AsrSession for WxasrStream {
    fn push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
        let Some(stream) = self.stream.as_mut() else {
            return;
        };
        // 调用方格式 → 16 kHz 单声道;攒重采样块时可能暂无输出
        let converted = self.convert.push(pcm);
        if converted.is_empty() {
            return;
        }
        let pcm: &[i16] = &converted;
        let ms = pcm.len() as u32 / 16; // 16000 Hz → 16 样本/ms
        self.quiet_ms = if self.vad.is_speech(pcm) {
            0
        } else {
            self.quiet_ms + ms
        };
        stream.push(pcm);
        self.stream_ms += ms;

        // 中间结果:新增够多、且距上次间隔不少于上次计算耗时的 3 倍
        if self.partial
            && stream.fresh() >= self.partial_min_fresh
            && self.last_emit.elapsed() >= self.last_cost * 3
        {
            let t0 = Instant::now();
            let p = stream.partial();
            self.last_cost = t0.elapsed();
            self.last_emit = Instant::now();
            if !p.is_empty() && p != self.last_partial {
                self.last_partial = p.clone();
                if !out.emit(AsrEvent::Partial(AsrText::plain(p))) {
                    return; // 消费端已关闭
                }
            }
        }

        // 上下文轮换:定稿当前流并接续(携带切点前 1s 左文与切点后未
        // 定稿尾部)。标点触发时切点优先取标点处,Final 以标点收尾
        if self.stream_ms > 0 {
            let ctx = RotateCtx {
                stream_ms: self.stream_ms,
                silence_ms: self.quiet_ms,
                punct_ms: stream.punct_ms(),
            };
            if self.rotate.should_rotate(&ctx) {
                let stream = self.stream.take().expect("push 开头已检查");
                let (text, next) = if self.rotate.punct_hit(&ctx) {
                    stream.rotate_at_punct()
                } else {
                    stream.rotate()
                };
                self.stream = Some(next);
                self.stream_ms = 0;
                self.last_partial.clear();
                if !text.is_empty() {
                    // 消费端已关闭(返回 false)也到此为止——这已是 push 的
                    // 最后一步;旧版 worker 此处 return,效果相同
                    let _ = out.emit(AsrEvent::Commit(AsrText::plain(text)));
                }
            }
        }
    }

    fn finish(&mut self, out: &mut dyn AsrEvents) {
        // 重采样残留先送进流,再定稿(不再参与轮换判定)
        let tail = self.convert.flush();
        if !tail.is_empty() {
            if let Some(stream) = self.stream.as_mut() {
                stream.push(&tail);
            }
        }
        let Some(stream) = self.stream.take() else {
            return;
        };
        let text = stream.finish();
        if !text.is_empty() {
            let _ = out.emit(AsrEvent::End(AsrText::plain(text))); // 消费端已关闭也忽略
        }
    }
}

/// 轻量自适应 VAD(能量门)。
///
/// 判定用 25ms 子块的 RMS 峰值:语音有音节起伏(峰值高),平稳底噪
/// 无论多响峰值都接近均值,因此比整块均值更能区分「软语音 vs 响底噪」。
/// 语音判定:峰值 > 噪声底×2.5 + 150(i16 幅度尺度)。
/// 噪声底只在静音段更新,且非对称:快速下跟、极慢上漂——防止被
/// 误判的软语音棘轮式抬高阈值。
///
/// wxasr 后端直接使用;gguf 后端在 Silero VAD 缺失或失败时降级用它。
#[derive(Default)]
pub(crate) struct Vad {
    noise: Option<f32>,
}

impl Vad {
    fn rms(pcm: &[i16]) -> f32 {
        if pcm.is_empty() {
            return 0.0;
        }
        (pcm.iter().map(|&v| v as f32 * v as f32).sum::<f32>() / pcm.len() as f32).sqrt()
    }

    pub(crate) fn is_speech(&mut self, pcm: &[i16]) -> bool {
        // 25ms 子块 RMS 峰值
        let peak = pcm.chunks(400).map(Self::rms).fold(0.0f32, f32::max);
        let avg = Self::rms(pcm);
        let noise = *self.noise.get_or_insert(avg);
        let speech = peak > noise * 2.5 + 150.0;
        if !speech {
            self.noise = Some(if avg < noise {
                noise * 0.8 + avg * 0.2 // 环境变安静:快速下跟
            } else {
                noise * 0.999 + avg * 0.001 // 环境变嘈杂:极慢上漂
            });
        }
        speech
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, amp: f64) -> Vec<i16> {
        (0..n)
            .map(|i| (amp * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin()) as i16)
            .collect()
    }

    #[test]
    fn vad_distinguishes_silence_and_speech() {
        let mut vad = Vad::default();
        // 数字静音:不是语音,且把噪声底拉低
        assert!(!vad.is_speech(&vec![0i16; 1600]));
        // 440Hz 中等音量:语音
        assert!(vad.is_speech(&tone(1600, 3000.0)));
        // 静音后小幅底噪仍不是语音
        assert!(!vad.is_speech(&vec![0i16; 1600]));
        let quiet = tone(1600, 30.0);
        assert!(!vad.is_speech(&quiet), "噪声底应自适应到小幅信号");
    }

    #[test]
    fn vad_uses_peak_not_average() {
        // 平稳信号(如空调/TTS 嗡声):子块峰值≈均值,无论多响都不是语音
        let mut vad = Vad::default();
        let hum = tone(6400, 1000.0); // RMS ≈ 707
        assert!(!vad.is_speech(&hum));
        assert!(!vad.is_speech(&hum));
        // 音节式信号:平均响度与底噪相当,但 25ms 峰值远超阈值 → 语音
        let syllabic: Vec<i16> = (0..6400)
            .map(|i| {
                let amp = if (i / 400) % 2 == 0 { 3000.0 } else { 200.0 };
                (amp * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin()) as i16
            })
            .collect();
        assert!(vad.is_speech(&syllabic));
    }

    #[test]
    fn vad_initializes_from_loud_environment() {
        // 首块就是大底噪(如空调声):讲话仍可判语音
        let mut vad = Vad::default();
        let hum = tone(6400, 700.0);
        assert!(!vad.is_speech(&hum));
        assert!(!vad.is_speech(&vec![0i16; 1600]));
        assert!(vad.is_speech(&tone(6400, 6000.0)));
    }
}
