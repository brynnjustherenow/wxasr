//! 桌面音频采集 → 识别 → 每段定稿一行写入 txt。
//!
//! 用法：
//! ```
//! cargo run --release --example desktop_txt -- <模型目录> <out.txt> [--play a.wav ...] [--secs 15]
//! ```
//!
//! `--play` 会把给定 wav 播放到系统输出（同时被 loopback 采集），
//! 便于自测：`--play` 且未指定 `--secs` 时自动按播放时长 + 3s 结束。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wxasr::Model;
use wxasr_async::{start, AsrEvent, Config};
use wxasr_provider::{AudioSource, DesktopAudio, Resampler};

struct Player {
    _stream: cpal::Stream,
    cursor: Arc<AtomicUsize>,
    total: usize,
    rate: u32,
}

/// 把 wav 列表重采样到输出设备格式并播放一遍（f32 输出设备）
fn spawn_player(wavs: &[String]) -> Option<Player> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    let host = cpal::default_host();
    let dev = host.default_output_device()?;
    let cfg = dev.default_output_config().ok()?;
    if cfg.sample_format() != cpal::SampleFormat::F32 {
        eprintln!("[提示] 输出设备非 f32 格式，跳过播放");
        return None;
    }
    let (rate, ch) = (cfg.sample_rate().0, cfg.channels() as usize);
    let mut mono: Vec<f32> = Vec::new();
    for w in wavs {
        match hound::WavReader::open(w) {
            Ok(mut r) => {
                let wav_rate = r.spec().sample_rate;
                let samples: Vec<f32> =
                    r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
                let mut out = Vec::new();
                Resampler::new(wav_rate, rate).push(&samples, &mut out);
                mono.extend(out.iter().map(|&v| v as f32 / 32767.0));
            }
            Err(e) => eprintln!("[提示] 跳过 {w}: {e}"),
        }
    }
    let total = mono.len();
    let data = Arc::new(mono);
    let cursor = Arc::new(AtomicUsize::new(0));
    let c = cursor.clone();
    let stream = dev
        .build_output_stream(
            &cfg.into(),
            move |d: &mut [f32], _| {
                let mut pos = c.load(Ordering::Relaxed);
                for frame in d.chunks_mut(ch) {
                    let s = if pos < data.len() { data[pos] } else { 0.0 };
                    pos += 1;
                    for v in frame.iter_mut() {
                        *v = s;
                    }
                }
                c.store(pos, Ordering::Relaxed);
            },
            |e| eprintln!("播放错误: {e}"),
            None,
        )
        .ok()?;
    stream.play().ok()?;
    Some(Player { _stream: stream, cursor, total, rate })
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir = String::from("G:/codes/rust/project-a/model");
    let mut out_path = String::from("out.txt");
    let mut wavs = Vec::new();
    let mut secs: Option<u64> = None;
    let mut pos = args.iter();
    if let Some(a) = pos.next() {
        model_dir = a.clone();
    }
    if let Some(a) = pos.next() {
        out_path = a.clone();
    }
    while let Some(a) = pos.next() {
        match a.as_str() {
            "--play" => {
                while let Some(w) = pos.next() {
                    if w.starts_with("--") {
                        break;
                    }
                    wavs.push(w.clone());
                }
            }
            "--secs" => {
                secs = pos.next().and_then(|s| s.parse().ok());
            }
            other => eprintln!("[提示] 未知参数: {other}"),
        }
    }

    let model = Model::load(&model_dir).expect("模型加载失败");
    let (asr, mut rx) = start(model, Config::default());
    let sink: wxasr_provider::PcmSink = {
        let asr = asr.clone();
        Arc::new(move |pcm: &[i16]| asr.push(pcm))
    };

    let source = DesktopAudio::new();
    println!("桌面设备: {:?}", source.devices().unwrap_or_default());
    let mut capture = source.start(None, sink).expect("桌面采集启动失败");

    let player = (!wavs.is_empty()).then(|| spawn_player(&wavs)).flatten();
    let secs = secs.unwrap_or_else(|| {
        player
            .as_ref()
            .map_or(15, |p| p.total as u64 / p.rate as u64 + 3)
    });

    println!("采集中 {secs}s（每段定稿自动分行）…");
    let t0 = Instant::now();
    // 每段定稿一行；Partial 是当前段的中间结果
    let mut lines: Vec<String> = Vec::new();
    let mut stopped = false;
    let mut asr_fin = Some(asr);
    loop {
        match tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
            Ok(Some(ev)) => match ev {
                AsrEvent::Partial(p) => println!("… {p}"),
                AsrEvent::Final(f) => {
                    println!("段: {f}");
                    lines.push(f);
                }
            },
            Ok(None) => break, // 通道关闭：会话结束
            Err(_) => {}     // 超时继续轮询
        }
        if !stopped && t0.elapsed().as_secs() >= secs {
            capture.stop();
            if let Some(a) = asr_fin.take() {
                a.finish();
            }
            stopped = true;
        }
    }
    capture.stop();
    std::fs::write(&out_path, lines.join("\n") + "\n").expect("写文件失败");
    println!("\n共 {} 段，已写入 {out_path}", lines.len());
}
