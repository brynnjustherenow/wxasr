//! wxasr-async —— [wxasr](../wxasr) 的异步事件流封装。
//!
//! 结构：一个专用识别线程拥有 [`wxasr::Stream`]（CPU 密集、有状态，
//! 不适合放进 async 执行器），负责中间结果节流与静音断句；事件经
//! tokio mpsc 桥接到 async 侧，async 只做分发。音频喂入是同步接口，
//! 可直接在采集回调里调用。
//!
//! # 事件流
//!
//! - [`AsrEvent::Partial`]：中间结果，已按计算成本自动节流（新增 ≥
//!   `partial_min_fresh` 帧，且距上次 ≥ 3× 上次计算耗时，慢机自动降频）
//! - [`AsrEvent::Sentence`]：静音超过 `silence_ms` 自动断句定稿
//! - [`AsrEvent::Final`]：[`AsrHandle::finish`] 或所有句柄 drop 后的
//!   最后一段定稿，是关闭事件
//!
//! # 入门
//!
//! ```no_run
//! # async fn demo(model: std::sync::Arc<wxasr::Model>, mic_pcm: &[i16]) {
//! use wxasr_async::{start, AsrEvent};
//!
//! let (handle, mut rx) = start(model, Default::default());
//! handle.push(mic_pcm);           // 采集回调里喂 16kHz 单声道 i16
//! handle.finish();                // 结束识别
//!
//! while let Some(ev) = rx.recv().await {
//!     match ev {
//!         AsrEvent::Partial(t) => { /* 边说边显 */ }
//!         AsrEvent::Sentence(t) => { /* 断句上屏 */ }
//!         AsrEvent::Final(t) => { println!("{t}"); break; }
//!     }
//! }
//! # }
//! ```
//!
//! # 静音检测
//!
//! 内置轻量自适应 VAD：噪声底取静音段 RMS 的滑动均值（初值为首个
//! 块的 RMS），判定 `rms > max(噪声底×4, 80)`。数字静音与不同增益的
//! 麦克风均可工作；注意初值取自首个块，若开口极快请先喂一小段环境音。

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use wxasr::Model;

/// 识别事件（按时间顺序产生）
#[derive(Debug, Clone)]
pub enum AsrEvent {
    /// 中间结果（已节流，仅当开启 `Config::partial`）
    Partial(String),
    /// 静音断句定稿
    Sentence(String),
    /// 结束后的最后一段定稿；收到它之后通道将关闭
    Final(String),
}

/// 运行配置
#[derive(Debug, Clone)]
pub struct Config {
    /// 静音断句窗口（毫秒），0 = 不自动断句
    pub silence_ms: u32,
    /// 是否产生 [`AsrEvent::Partial`]
    pub partial: bool,
    /// 中间结果的最小新增帧数（10ms/帧）
    pub partial_min_fresh: usize,
    /// 事件通道容量（消费方卡死时识别线程最多积压这么多事件）
    pub event_capacity: usize,
    /// 识别线程名。`None` 时自动生成 `wxasr-<uuidv7>`（时间有序，天然唯一），
    /// 多流并发无需配置也不会重名；需要可读名字（如 "wxasr-desktop"）时显式传入。
    pub thread_name: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            silence_ms: 800,
            partial: true,
            partial_min_fresh: 30,
            event_capacity: 64,
            thread_name: None,
        }
    }
}

enum AudioMsg {
    Pcm(Vec<i16>),
    End,
}

/// 音频喂入端。`Clone` 后可多处持有（如采集回调与停止按钮）；
/// 任一处调用 [`finish`](Self::finish) 即结束识别，之后的 `push` 静默忽略。
#[derive(Clone)]
pub struct AsrHandle {
    tx: std::sync::mpsc::Sender<AudioMsg>,
}

impl AsrHandle {
    /// 喂入一段 16kHz 单声道 i16 PCM（任意块大小）。
    /// 同步、非阻塞、不丢数据，可直接在采集回调里调用。
    pub fn push(&self, pcm: &[i16]) {
        let _ = self.tx.send(AudioMsg::Pcm(pcm.to_vec()));
    }

    /// 结束识别：识别线程定稿最后一段并发送 [`AsrEvent::Final`]。
    /// 消耗 self；所有句柄 drop 时同样触发结束。
    pub fn finish(self) {
        let _ = self.tx.send(AudioMsg::End);
    }
}

/// 启动识别。返回（喂入端，事件流）。
///
/// 多个识别流可共享同一 `Arc<Model>`：权重与内部计算线程池全局复用，
/// 每流仅一个 worker 线程与各自的流式状态（特征/VAD）。并发流建议
/// 通过 [`Config::thread_name`] 各自命名。
pub fn start(model: Arc<Model>, cfg: Config) -> (AsrHandle, mpsc::Receiver<AsrEvent>) {
    let (audio_tx, audio_rx) = std::sync::mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel(cfg.event_capacity);
    let name = match cfg.thread_name.clone() {
        Some(n) => n,
        // uuidv7：时间有序、并发不重名（Linux 线程名 15 字节截断不影响前缀唯一性）
        None => format!("wxasr-{}", uuid::Uuid::now_v7()),
    };
    std::thread::Builder::new()
        .name(name)
        .spawn(move || worker(model, cfg, audio_rx, event_tx))
        .expect("识别线程创建失败");
    (AsrHandle { tx: audio_tx }, event_rx)
}

fn worker(
    model: Arc<Model>,
    cfg: Config,
    audio: std::sync::mpsc::Receiver<AudioMsg>,
    out: mpsc::Sender<AsrEvent>,
) {
    let mut stream = model.stream();
    let mut vad = Vad::default();
    let mut quiet_ms = 0u32;
    let mut last_emit = Instant::now();
    let mut last_cost = Duration::ZERO;
    let mut last_partial = String::new();

    while let Ok(msg) = audio.recv() {
        let AudioMsg::Pcm(pcm) = msg else { break };
        let ms = pcm.len() as u32 / 16; // 16000 Hz → 16 样本/ms
        quiet_ms = if vad.is_speech(&pcm) {
            0
        } else {
            quiet_ms + ms
        };
        stream.push(&pcm);

        // 中间结果：新增够多、且距上次间隔不少于上次计算耗时的 3 倍
        if cfg.partial
            && stream.fresh() >= cfg.partial_min_fresh
            && last_emit.elapsed() >= last_cost * 3
        {
            let t0 = Instant::now();
            let p = stream.partial();
            last_cost = t0.elapsed();
            last_emit = Instant::now();
            if !p.is_empty() && p != last_partial {
                last_partial = p.clone();
                if out.blocking_send(AsrEvent::Partial(p)).is_err() {
                    return; // 消费端已关闭
                }
            }
        }

        // 静音断句：定稿当前句，换新 Stream（权重共享，重建的只是特征状态）
        if cfg.silence_ms > 0 && quiet_ms >= cfg.silence_ms {
            let text = stream.finish();
            stream = model.stream();
            quiet_ms = 0;
            last_partial.clear();
            if !text.is_empty() && out.blocking_send(AsrEvent::Sentence(text)).is_err() {
                return;
            }
        }
    }
    let _ = out.blocking_send(AsrEvent::Final(stream.finish()));
}

/// 轻量自适应 VAD。
///
/// 判定用 25ms 子块的 RMS 峰值：语音有音节起伏（峰值高），平稳底噪
/// 无论多响峰值都接近均值，因此比整块均值更能区分「软语音 vs 响底噪」。
/// 语音判定：峰值 > 噪声底×2.5 + 150（i16 幅度尺度）。
/// 噪声底只在静音段更新，且非对称：快速下跟、极慢上漂——防止被
/// 误判的软语音棘轮式抬高阈值。
struct Vad {
    noise: Option<f32>,
}

impl Default for Vad {
    fn default() -> Self {
        Vad { noise: None }
    }
}

impl Vad {
    fn rms(pcm: &[i16]) -> f32 {
        if pcm.is_empty() {
            return 0.0;
        }
        (pcm.iter().map(|&v| v as f32 * v as f32).sum::<f32>() / pcm.len() as f32).sqrt()
    }

    fn is_speech(&mut self, pcm: &[i16]) -> bool {
        // 25ms 子块 RMS 峰值
        let peak = pcm.chunks(400).map(Self::rms).fold(0.0f32, f32::max);
        let avg = Self::rms(pcm);
        let noise = *self.noise.get_or_insert(avg);
        let speech = peak > noise * 2.5 + 150.0;
        if !speech {
            self.noise = Some(if avg < noise {
                noise * 0.8 + avg * 0.2 // 环境变安静：快速下跟
            } else {
                noise * 0.999 + avg * 0.001 // 环境变嘈杂：极慢上漂
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
        // 数字静音：不是语音，且把噪声底拉低
        assert!(!vad.is_speech(&vec![0i16; 1600]));
        // 440Hz 中等音量：语音
        assert!(vad.is_speech(&tone(1600, 3000.0)));
        // 静音后小幅底噪仍不是语音
        assert!(!vad.is_speech(&vec![0i16; 1600]));
        let quiet = tone(1600, 30.0);
        assert!(!vad.is_speech(&quiet), "噪声底应自适应到小幅信号");
    }

    #[test]
    fn vad_uses_peak_not_average() {
        // 平稳信号（如空调/TTS 嗡声）：子块峰值≈均值，无论多响都不是语音
        let mut vad = Vad::default();
        let hum = tone(6400, 1000.0); // RMS ≈ 707
        assert!(!vad.is_speech(&hum));
        assert!(!vad.is_speech(&hum));
        // 音节式信号：平均响度与底噪相当，但 25ms 峰值远超阈值 → 语音
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
        // 首块就是大底噪（如空调声）：讲话仍可判语音
        let mut vad = Vad::default();
        let hum = tone(6400, 700.0);
        assert!(!vad.is_speech(&hum));
        assert!(!vad.is_speech(&vec![0i16; 1600]));
        assert!(vad.is_speech(&tone(6400, 6000.0)));
    }
}
