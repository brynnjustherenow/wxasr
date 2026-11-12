//! 错误类型

/// 库错误
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// 文件 IO 错误
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// 模型内容问题：缺少张量、形状不符、词表不匹配等
    #[error("model: {0}")]
    Model(String),
    /// `.xnet` 容器解析失败
    #[error("xnet 容器: {0}")]
    Container(String),
    /// 音频格式不支持（只支持 16kHz 单声道 16bit PCM）
    #[error("audio: {0}")]
    Audio(String),
}

/// 库结果类型
pub type Result<T> = std::result::Result<T, Error>;
