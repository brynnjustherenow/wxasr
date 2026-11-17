//! hex-asr — standalone GGUF speech-to-text, extracted from Hex's
//! Windows/Linux inference stack.
//!
//! The layering is Hex's: [`engine`] (one dedicated thread; ordered command
//! queue, dictation foreground vs. background priority, a 1.5 s
//! stream-finish deadline that falls back to the whole clip, job deadlines,
//! bounded waits) → [`local_transcriber`] (GGUF load, device choice,
//! CPU recovery) → [`gguf_session`] (the transcribe-cpp session owner) →
//! transcribe-cpp (FFI).
//!
//! Audio is 16 kHz mono, internally f32. Two entry shapes:
//!
//! - Raw f32 ([`Engine`]/[`LiveStream`]), Hex's own surface: feed f32, take
//!   [`Partial`]s, finish with the authoritative clip.
//! - Push-based i16 ([`StreamingSession`]), for a unified async layer's
//!   session contract (wxasr-async's `AsrSession`): push `&[i16]` chunks,
//!   receive [`SessionEvent`]s through a [`SessionEvents`] sink, one
//!   `Final` per finish, no rotation. This crate adapts to that contract;
//!   the unified layer changes nothing on its side. i16→f32 conversion is
//!   [`audio::i16_to_f32`].
//!
//! ```no_run
//! # use hex_asr::{Device, Engine, ModelSpec};
//! # fn main() -> eyre::Result<()> {
//! let spec = ModelSpec::unified_english("parakeet-unified-en.gguf");
//! let engine = Engine::load(&spec, Device::Auto)?;
//!
//! // Stream while recording, then finish with the complete clip.
//! let mut stream = engine.begin().unwrap();      // None when it can't stream
//! stream.feed(&[0.0_f32; 1600]);                  // 100 ms of 16 kHz f32
//! if let Some(partial) = stream.take_partial() {  // speculative text
//!     println!("{}", partial.stable);
//! }
//! let whole_clip = [0.0_f32; 48_000].to_vec();    // the authoritative clip
//! let transcribed = engine
//!     .finish(Some(stream), whole_clip)?
//!     .wait()
//!     .map_err(|reason| eyre::eyre!("{reason}"))?;
//! println!("{}", transcribed.text);
//!
//! // Or transcribe a whole clip at once.
//! let transcribed = engine.transcribe_clip(vec![0.0_f32; 48_000])?.wait().unwrap();
//! # let _ = transcribed;
//! # Ok(())
//! # }
//! ```

pub mod audio;
pub mod engine;
pub mod gguf_session;
pub mod gpu_health;
pub mod local_transcriber;
pub mod pause_segmenter;
pub mod session;
pub mod speech_detector;
pub mod subtitle_credits;
pub mod transcript;

pub use audio::{Resampler, SAMPLE_RATE};
pub use engine::{
    Engine, LiveStream, Offline, Partial, PendingTranscript, Source, Transcribed, YieldingStream,
};
pub use local_transcriber::{Device, LocalTranscriber, ModelSpec};
pub use session::{SessionEvent, SessionEvents, StreamingSession};
pub use transcript::{TimestampLevel, Transcript, TranscriptSegment};

/// Silence after a Unified English clip (400 ms at 16 kHz), so the model
/// has the context to emit the final word and its closing punctuation
/// (issue 657: sentences lost their full stop or question mark).
pub const UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES: usize = 6_400;

/// Pads a GGUF clip as dictation prepares it: the minimum duration, then
/// the model's trailing silence. Applied after any pause cut, so a
/// trailing pause cannot trim that silence.
pub fn pad_gguf_samples(samples: &mut Vec<f32>, trailing_silence: usize) {
    audio::pad_minimum(samples);
    samples.resize(samples.len() + trailing_silence, 0.0);
}
