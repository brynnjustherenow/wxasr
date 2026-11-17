//! hex-asr 后端:GGUF 推理栈(transcribe-cpp,whisper/Qwen 等模型家族)
//! 接入统一层。
//!
//! [`HexBackend`] 持有一个已加载的 [`hex_asr::Engine`](共享的专用推理
//! 线程,模型常驻,多次会话零重复加载);每次 [`open`](AsrBackend::open)
//! 在其上开一个 [`StreamingSession`](hex_asr::StreamingSession) 会话。
//! 会话语义是 Hex 的:投机流式 partial、finish 携带完整 clip 兜底、
//! **一次会话恰一个 `Final`**(无上下文轮换——多段 `Final` 是 wxasr
//! 后端自己的行为)。`Config::rotate`/`partial_min_fresh` 被忽略,
//! `Config::partial` 控制是否发中间结果。
//!
//! 富字段直通:[`AsrText::language`] 来自模型自报的语言识别
//! (whisper 等),词/段时间戳字段在引擎以时间戳级别运行时填充
//! (默认整段路径不带时间戳,字段为空,见
//! [`hex_asr::Transcript`])。

use std::sync::Arc;

use crate::preprocess::InputConverter;
use crate::{
    AsrBackend, AsrEvent, AsrEvents, AsrSegment, AsrSession, AsrText, AsrWord, AudioIn, Config,
};
use hex_asr::{Engine, ModelSpec, SessionEvent, StreamingSession, Transcript};

/// hex-asr 后端:封装一个已加载的引擎。
///
/// 引擎的加载(含模型加载与设备选择)是重操作,由调用方完成并共享:
/// `HexBackend::new(Arc<Engine>)`;或用 [`HexBackend::from_spec`] 一步
/// 加载。多个并发会话共享同一引擎(引擎内部按序调度)。
pub struct HexBackend {
    engine: Arc<Engine>,
}

impl HexBackend {
    /// 用已加载的引擎创建(推荐:调用方控制加载时机与设备)。
    pub fn new(engine: Arc<Engine>) -> Self {
        HexBackend { engine }
    }

    /// 加载并创建。`Engine::load` 的直通(模型加载、设备选择、预热)。
    pub fn from_spec(spec: &ModelSpec, device: hex_asr::Device) -> eyre::Result<Self> {
        Ok(HexBackend {
            engine: Arc::new(Engine::load(spec, device)?),
        })
    }
}

impl AsrBackend for HexBackend {
    fn open(&self, cfg: &Config, audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>> {
        // hex-asr 的会话吃 16 kHz 单声道 i16:调用方声明的格式在这里转换,
        // 16 kHz 单声道直通。
        let convert = InputConverter::new(audio)
            .map_err(|reason| eyre::eyre!("hex-asr 不支持该音频格式: {reason}"))?;
        Ok(Box::new(HexStream {
            session: StreamingSession::open(self.engine.clone(), cfg.partial),
            convert,
        }))
    }
}

/// hex-asr 会话:一个 [`StreamingSession`] 加输入格式转换。
pub struct HexStream {
    session: StreamingSession,
    convert: InputConverter,
}

impl AsrSession for HexStream {
    fn push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
        let converted = self.convert.push(pcm);
        if converted.is_empty() {
            return;
        }
        self.session.push(&converted, &mut Bridge { out });
    }

    fn finish(&mut self, out: &mut dyn AsrEvents) {
        // 重采样残留先进会话,再定稿
        let tail = self.convert.flush();
        if !tail.is_empty() {
            self.session.push(&tail, &mut Bridge { out });
        }
        self.session.finish(&mut Bridge { out });
    }
}

/// hex-asr 的 [`SessionEvent`] → 统一层的 [`AsrEvent`]。
struct Bridge<'a> {
    out: &'a mut dyn AsrEvents,
}

impl hex_asr::SessionEvents for Bridge<'_> {
    fn emit(&mut self, event: SessionEvent) -> bool {
        match event {
            SessionEvent::Partial(text) => self.out.emit(AsrEvent::Partial(AsrText::plain(text))),
            SessionEvent::Final(transcript) => self.out.emit(AsrEvent::Final(asr_text(transcript))),
        }
    }
}

/// [`hex_asr::Transcript`] → [`AsrText`]:富字段直通,时间戳毫秒原样。
fn asr_text(transcript: Transcript) -> AsrText {
    // 整段说话人:所有分段归属同一个说话人时才有,否则 None
    let speaker = transcript
        .segments
        .iter()
        .map(|segment| segment.speaker)
        .reduce(|acc, speaker| match (acc, speaker) {
            (Some(a), Some(b)) if a == b => Some(a),
            _ => None,
        })
        .flatten();
    AsrText {
        content: transcript.text,
        language: transcript.language,
        speaker,
        segments: transcript
            .segments
            .into_iter()
            .map(|segment| AsrSegment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text,
                speaker: segment.speaker,
            })
            .collect(),
        words: transcript
            .words
            .into_iter()
            .map(|word| AsrWord {
                start_ms: word.start_ms,
                end_ms: word.end_ms,
                text: word.text,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timed_transcript_maps_onto_the_rich_event() {
        let transcript = Transcript {
            text: "hello world".into(),
            language: Some("en".into()),
            segments: vec![
                hex_asr::TranscriptSegment {
                    start_ms: 0,
                    end_ms: 500,
                    text: "hello".into(),
                    speaker: Some(1),
                },
                hex_asr::TranscriptSegment {
                    start_ms: 500,
                    end_ms: 900,
                    text: "world".into(),
                    speaker: Some(1),
                },
            ],
            words: vec![hex_asr::TranscriptSegment {
                start_ms: 0,
                end_ms: 300,
                text: "hello".into(),
                speaker: None,
            }],
        };
        let text = asr_text(transcript);
        assert_eq!(text.content, "hello world");
        assert_eq!(text.language.as_deref(), Some("en"));
        assert_eq!(text.speaker, Some(1), "统一归属的说话人上浮到整段");
        assert_eq!(text.segments.len(), 2);
        assert_eq!(text.segments[0].speaker, Some(1));
        assert_eq!(text.words.len(), 1);
        assert_eq!(text.words[0].end_ms, 300);
    }

    #[test]
    fn mixed_speakers_do_not_surface_a_span_speaker() {
        let transcript = Transcript {
            text: "a b".into(),
            language: None,
            segments: vec![
                hex_asr::TranscriptSegment {
                    start_ms: 0,
                    end_ms: 100,
                    text: "a".into(),
                    speaker: Some(1),
                },
                hex_asr::TranscriptSegment {
                    start_ms: 100,
                    end_ms: 200,
                    text: "b".into(),
                    speaker: Some(2),
                },
            ],
            words: Vec::new(),
        };
        let text = asr_text(transcript);
        assert_eq!(text.speaker, None, "混合说话人不上浮");
        assert_eq!(text.segments[0].speaker, Some(1));
        assert_eq!(text.segments[1].speaker, Some(2));
    }
}
