//! Streaming Silero voice-activity detection: a 1.2 MB GGUF scored on the
//! CPU, giving a recognition backend the speech/silence signal an endpointing
//! rotation policy needs (the silence half of the unified layer's
//! `RotateCtx`).
//!
//! [`SileroVad`] wraps one [`transcribe_cpp::VadSession`] plus the state an
//! endpointer reads off it: the latest frame's speech probability and the
//! current run of trailing silence. Audio is 16 kHz mono f32 in any chunk
//! size — the native session buffers partial frames (512 samples = 32 ms
//! each) itself. It runs on the caller's thread, not the engine thread: a
//! frame is sub-millisecond on the CPU.
//!
//! Load the model once ([`SileroVad::load`]) and [`fork`](SileroVad::fork) a
//! VAD per recognition session: the stream position and silence run are
//! per-session state and must not bleed across sessions. If a GPU engine
//! shares the process, load it first — the first
//! [`local_transcriber::init_backends`] call pins the process's backends, and
//! a VAD loaded first would pin them to CPU-only.

use std::path::Path;

use eyre::{Context, Result};

/// A frame at or above this probability is speech (Silero's default).
const THRESHOLD: f32 = 0.5;
/// Below this a frame is silence: Silero's `neg_threshold` default,
/// `threshold - 0.15`. Between the two the frame is neither — too quiet to
/// be speech, too loud to keep a silence run going — so it clears the run.
const SILENCE_PROB: f32 = 0.35;
/// One scored frame: 512 samples at 16 kHz.
const FRAME_MS: u32 = 32;

/// The endpointer state carried between frames. Pure, so tests inject
/// probabilities directly instead of loading a model.
#[derive(Default)]
struct SilenceState {
    /// The latest completed frame's speech probability.
    prob: Option<f32>,
    /// The current run of consecutive silent frames, in milliseconds.
    silence_ms: u32,
}

impl SilenceState {
    /// Advances over completed frames, oldest first.
    fn on_probs(&mut self, probs: &[f32]) {
        for &prob in probs {
            self.prob = Some(prob);
            self.silence_ms = if prob < SILENCE_PROB {
                self.silence_ms.saturating_add(FRAME_MS)
            } else {
                0
            };
        }
    }

    /// Whether the latest completed frame is speech.
    fn is_speech(&self) -> bool {
        self.prob.is_some_and(|prob| prob >= THRESHOLD)
    }
}

/// One streaming Silero VAD: feed 16 kHz mono f32, read speech/silence.
/// Session-scoped — [`fork`](Self::fork) one per recognition session; never
/// share a VAD across sessions.
pub struct SileroVad {
    /// Kept so forks open new native sessions without re-reading the file.
    model: transcribe_cpp::Model,
    session: transcribe_cpp::VadSession,
    state: SilenceState,
}

impl SileroVad {
    /// Loads the Silero GGUF on the CPU (1.2 MB, tens of milliseconds). If a
    /// GPU engine shares the process, load that first (see the module docs).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // Model::load_with would otherwise register every backend unguarded
        // (Hex #662); CPU is all a VAD needs.
        crate::local_transcriber::init_backends(crate::local_transcriber::Device::Cpu)?;
        let model = transcribe_cpp::Model::load_with(
            path,
            &transcribe_cpp::ModelOptions {
                backend: transcribe_cpp::Backend::Cpu,
                ..Default::default()
            },
        )
        .with_context(|| format!("could not load the VAD model from {}", path.display()))?;
        let session = model.vad_session()?;
        Ok(Self {
            model,
            session,
            state: SilenceState::default(),
        })
    }

    /// A fresh VAD over the same loaded model: its own native session and
    /// state, so concurrent recognition sessions never read each other's
    /// audio. Does not read the model file again.
    pub fn fork(&self) -> Result<Self> {
        Ok(Self {
            model: self.model.clone(),
            session: self.model.vad_session()?,
            state: SilenceState::default(),
        })
    }

    /// Appends 16 kHz mono f32 of any chunk size, scoring every frame it
    /// completes (a partial frame waits inside the native session).
    pub fn feed(&mut self, pcm: &[f32]) -> Result<()> {
        if pcm.is_empty() {
            return Ok(());
        }
        let scored = self.session.stream_feed(pcm)?;
        self.state.on_probs(&scored.probs);
        Ok(())
    }

    /// Whether the latest completed frame is speech (p ≥ 0.5). `false`
    /// before the first frame.
    pub fn is_speech(&self) -> bool {
        self.state.is_speech()
    }

    /// The current run of consecutive silence, in milliseconds: frames at
    /// p < 0.35 each add 32 ms, anything louder clears it (speech, and the
    /// hysteresis band under the threshold). 0 before the first frame.
    pub fn silence_ms(&self) -> u32 {
        self.state.silence_ms
    }

    /// Drops the scored stream; the next feed starts at frame 0. Reserved —
    /// rotation does not call it: silence describes the audio, not the
    /// stream, and stays across a rotation (the unified layer's
    /// `RotateCtx::silence_ms` semantics).
    pub fn reset(&mut self) {
        self.session.stream_reset();
        self.state = SilenceState::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_accumulates_per_frame_and_clears_on_speech() {
        let mut state = SilenceState::default();
        assert_eq!(state.silence_ms, 0);
        assert!(!state.is_speech(), "no frame yet is not speech");
        state.on_probs(&[0.1; 10]);
        assert_eq!(state.silence_ms, 320, "each silent frame adds 32 ms");
        state.on_probs(&[0.2, 0.1]);
        assert_eq!(state.silence_ms, 384);
        state.on_probs(&[0.9]);
        assert_eq!(state.silence_ms, 0, "speech clears the run");
        assert!(state.is_speech());
    }

    #[test]
    fn the_hysteresis_band_is_neither_speech_nor_silence() {
        let mut state = SilenceState::default();
        state.on_probs(&[0.1; 10]);
        // p in [0.35, 0.5): not speech, but it still clears the silence run.
        state.on_probs(&[0.4]);
        assert!(!state.is_speech());
        assert_eq!(state.silence_ms, 0);
        // The boundary itself counts as loud enough to clear.
        state.on_probs(&[0.1]);
        state.on_probs(&[0.35]);
        assert_eq!(state.silence_ms, 0);
        // Just under it is silence.
        state.on_probs(&[0.34]);
        assert_eq!(state.silence_ms, 32);
    }

    /// The VAD file to load, or `None` to skip: `HEXASR_VAD_MODEL` overrides
    /// the project's own copy in the models directory.
    fn vad_path() -> Option<std::path::PathBuf> {
        match std::env::var("HEXASR_VAD_MODEL") {
            Ok(path) if !path.is_empty() => Some(path.into()),
            _ => {
                let default =
                    std::path::PathBuf::from(r"G:\env\model\asr\silero-vad-v6.2-F32.gguf");
                default.is_file().then_some(default)
            }
        }
    }

    /// The real detector, gated on the model file existing (the pattern of
    /// `HEXASR_MODEL` in tests/model.rs, with the project's own copy as the
    /// default).
    #[test]
    fn the_real_detector_scores_silence_and_speech() {
        let Some(path) = vad_path() else {
            return;
        };
        let mut vad = SileroVad::load(&path).unwrap();

        // 2 s of digital silence: not speech, a ~2 s silence run (the
        // trailing partial frame stays unscored).
        vad.feed(&[0.0; 32_000]).unwrap();
        assert!(!vad.is_speech());
        assert!(
            (1_900..=2_048).contains(&vad.silence_ms()),
            "2 s of silence should run ~2 s, got {} ms",
            vad.silence_ms()
        );

        // Real speech clears it: the TTS fixture shared with wxasr's tests.
        let wav = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../wxasr/tests/data/tts_short_1.wav"
        );
        let mut reader = hound::WavReader::open(wav).expect("the shared TTS fixture");
        let speech: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32768.0)
            .collect();
        let mut saw_speech = false;
        for chunk in speech.chunks(1_600) {
            // 100 ms at a time
            vad.feed(chunk).unwrap();
            if vad.is_speech() {
                saw_speech = true;
                break;
            }
        }
        assert!(saw_speech, "the TTS clip contains speech");
        assert_eq!(vad.silence_ms(), 0, "speech clears the silence run");
    }

    /// A fork scores independently of its parent: the state never bleeds
    /// between two sessions sharing one loaded model.
    #[test]
    fn a_fork_starts_from_a_clean_state() {
        let Some(path) = vad_path() else {
            return;
        };
        let mut vad = SileroVad::load(&path).unwrap();
        vad.feed(&[0.0; 8_000]).unwrap(); // 500 ms: 15 frames + a partial
        assert_eq!(vad.silence_ms(), 480);
        let mut forked = vad.fork().unwrap();
        assert_eq!(forked.silence_ms(), 0, "a fork starts unscored");
        // The fork scores its own stream; the parent is untouched.
        forked.feed(&[0.0; 1_600]).unwrap(); // 100 ms: 3 frames
        assert_eq!(forked.silence_ms(), 96);
        assert_eq!(vad.silence_ms(), 480, "the parent kept its own run");
    }
}
