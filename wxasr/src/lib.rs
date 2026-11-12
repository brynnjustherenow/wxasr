//! wxasr —— 微信输入法离线语音识别模型（`.xnet`）的纯 Rust 推理库。
//!
//! 端到端流程：16kHz 单声道 PCM → 40 维 log-mel FBank（滑动均值归一化）
//! → 卷积前端（10ms 帧降为 50ms 帧）→ 40 层 int8 Transformer → CTC → 文本。
//! 无深度学习框架依赖；int8 动态量化 GEMM 自带 AVX2 内核。
//!
//! # 模型文件
//!
//! 从微信输入法官方语音包（双层 zip 的 apk）解出，目录内需包含：
//! - 任一 `*.xnet` 文件（权重容器）
//! - `dict.decoder.utf8.txt`（CTC 词表）
//!
//! # 入门
//!
//! ```no_run
//! # use wxasr::Result;
//! # fn main() -> Result<()> {
//! // 加载模型（返回 Arc，多个识别流可共享同一份）
//! let model = wxasr::Model::load("model")?;
//!
//! // 一次性转写整段 wav（16kHz 单声道 16bit）
//! let text = model.transcribe_wav("test.wav")?;
//!
//! // 流式识别
//! let mut s = model.stream();
//! let pcm_chunk: Vec<i16> = vec![0; 3200]; // 从采集设备读到的任意块
//! s.push(&pcm_chunk);
//! let partial = s.partial(); // 中间结果（调用方按 fresh() 节流）
//! let final_text = s.finish(); // 定稿
//! # Ok(())
//! # }
//! ```
//!
//! # 线程
//!
//! 前向计算使用内部线程池，封顶 4 线程（实测更多线程只会增加调度
//! 开销与整机占用）。[`Model`] 可跨线程共享。
//!
//! # 致谢
//!
//! 容器格式、特征参数与流式策略参考了
//! [J3n5en/voicekey](https://github.com/J3n5en/voicekey) 中 `wtlocal`
//! 的实现思路。该项目未附带开源许可证，故本库未复用其代码，
//! 全部为独立重实现。

pub mod xnet;

mod error;
mod fbank;
mod kernels;
mod model;
mod stream;
pub use error::{Error, Result};
pub use fbank::{Fbank, DIM};
pub use model::Model;
pub use stream::Stream;

use std::path::Path;

impl Model {
    /// 一次性转写整段 wav（整段解码，不走分块流式）。
    ///
    /// 仅支持 16kHz 单声道 16bit PCM wav；长音频建议用 [`Model::stream`]。
    pub fn transcribe_wav(&self, path: impl AsRef<Path>) -> Result<String> {
        let path = path.as_ref();
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
        let spec = reader.spec();
        if spec.sample_rate != 16000 || spec.channels != 1 || spec.bits_per_sample != 16 {
            return Err(Error::Audio(format!(
                "{}: 需要 16kHz/单声道/16bit，实际 {}Hz/{}ch/{}bit",
                path.display(),
                spec.sample_rate,
                spec.channels,
                spec.bits_per_sample
            )));
        }
        let pcm: Vec<i16> = reader
            .samples::<i16>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
        let mut fbank = Fbank::new();
        let mut frames = Vec::new();
        fbank.push(&pcm, &mut frames);
        Ok(crate::stream::tidy(&self.text(&self.forward(&frames))))
    }

    /// 基于该模型创建流式识别器（共享模型权重）
    pub fn stream(self: &std::sync::Arc<Self>) -> Stream {
        Stream::new(self.clone())
    }
}
