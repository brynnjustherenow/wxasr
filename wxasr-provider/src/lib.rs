//! wxasr-provider —— wxasr 的跨平台音频采集提供者。
//!
//! 统一把设备的原生格式（采样率/通道/样本类型）转换为 **16kHz 单声道
//! i16 PCM**（wxasr 的输入格式）后投递：
//! - [`Microphone`]：麦克风，全平台
//! - [`DesktopAudio`]：桌面回环（系统正在播放的声音）——Windows 用
//!   WASAPI loopback；Linux 用 PulseAudio/PipeWire 的 monitor 源；
//!   macOS 需 BlackHole 等虚拟设备（以输入设备形式出现）
//!
//! # 入门
//!
//! ```no_run
//! use std::sync::Arc;
//! use wxasr_provider::{AudioSource, DesktopAudio};
//!
//! let src = DesktopAudio::new();
//! println!("设备: {:?}", src.devices());
//! let mut capture = src.start(None, Arc::new(|pcm: &[i16]| {
//!     // 16kHz 单声道 i16，喂给 wxasr / wxasr-async
//! }))
//! .unwrap();
//! std::thread::sleep(std::time::Duration::from_secs(5));
//! capture.stop();
//! ```
//!
//! # 回调线程约定
//!
//! `PcmSink` 在音频线程上执行，必须快速返回——典型做法是只做一次
//! 通道转发（如 [`wxasr_async::AsrHandle::push`]），重活交给消费者。

mod cpal_impl;
mod resample;
mod source;

pub use cpal_impl::{DesktopAudio, Microphone};
pub use resample::Resampler;
pub use source::{AudioSource, Capture, Error, PcmSink, Result};
