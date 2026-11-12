//! 采集抽象：trait、会话句柄与错误类型。

use std::sync::Arc;
use thiserror::Error;

/// 采集错误
#[derive(Debug, Error)]
pub enum Error {
    /// 设备枚举/打开/配置失败
    #[error("设备错误: {0}")]
    Device(String),
    /// 平台或设备不支持该采集方式
    #[error("不支持: {0}")]
    Unsupported(String),
}

/// 采集结果
pub type Result<T> = std::result::Result<T, Error>;

/// PCM 接收端：16kHz 单声道 i16。
/// 回调在音频线程上执行，必须快速返回（典型做法是转发给通道）。
pub type PcmSink = Arc<dyn Fn(&[i16]) + Send + Sync>;

/// 音频源。实现方负责把设备的原生格式（采样率/通道/样本类型）
/// 转换为 16kHz 单声道 i16 后投递给 sink。
pub trait AudioSource: Send + Sync {
    /// 提供者名称（"microphone" / "desktop"）
    fn name(&self) -> &'static str;
    /// 列出可用设备名
    fn devices(&self) -> Result<Vec<String>>;
    /// 开始采集。`device` 为 None 用默认设备，Some(名称) 按名选择。
    /// 返回的 [`Capture`] drop 时自动停止。
    fn start(&self, device: Option<&str>, sink: PcmSink) -> Result<Capture>;
}

/// 采集会话。持有着底层音频流，[`stop`](Self::stop) 或 drop 时停止。
pub struct Capture {
    stream: Option<cpal::Stream>,
}

impl Capture {
    pub(crate) fn new(stream: cpal::Stream) -> Self {
        Self {
            stream: Some(stream),
        }
    }

    /// 停止采集（幂等）
    pub fn stop(&mut self) {
        self.stream = None;
    }

    /// 是否仍在采集
    pub fn is_running(&self) -> bool {
        self.stream.is_some()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop();
    }
}
