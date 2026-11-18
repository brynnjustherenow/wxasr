//! Silero voice activity detection (via transcribe-cpp's VAD session), used
//! to keep audio without speech away from the transcription models, which
//! tend to invent text for it.
//!
//! The detector is optional: point it at a Silero VAD GGUF with
//! [`set_model`] (or pass a path to [`load`]), and until then every caller
//! falls back to its level-only behavior. Unlike Hex, this build never
//! downloads anything: the caller owns the model file.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use eyre::Result;
use transcribe_cpp::{Backend, Model, ModelOptions, VadOptions, VadSegment, VadSession};

/// Speech shorter than this is ignored. Silero's default of 250 ms drops a
/// short "Ok" or "Ciao"; 160 ms keeps those and still rejects clicks, keys
/// and room noise.
const MIN_SPEECH_MS: i32 = 160;

/// Speech probability that counts as speech. Silero's default of 0.5 misses
/// a whisper at -43 dBFS, which a level gate deliberately keeps; 0.4 hears
/// it and still rejects the silence in a recorded meeting.
pub const THRESHOLD: f64 = 0.4;

/// Where the caller said the Silero VAD GGUF lives; `None` until set.
static MODEL: Mutex<Option<PathBuf>> = Mutex::new(None);

#[cfg_attr(test, allow(dead_code))]
enum State {
    Unloaded,
    Loaded(VadSession),
    Failed,
}

#[cfg_attr(test, allow(dead_code))]
static STATE: Mutex<State> = Mutex::new(State::Unloaded);
#[cfg_attr(test, allow(dead_code))]
static RUN_FAILURE_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Points the detector at a Silero VAD GGUF. Takes effect from the next
/// check; a session already loaded from an older path stays until then.
pub fn set_model(path: impl Into<PathBuf>) {
    let path = path.into();
    let mut state = lock_state();
    let mut model = lock_model();
    if model.as_ref() != Some(&path) {
        *state = State::Unloaded;
        *model = Some(path);
    }
}

/// The path the detector is pointed at, if any.
pub fn model_path() -> Option<PathBuf> {
    lock_model().clone()
}

fn lock_model() -> MutexGuard<'static, Option<PathBuf>> {
    MODEL.lock().unwrap_or_else(|error| error.into_inner())
}

fn lock_state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|error| error.into_inner())
}

/// Whether 16 kHz mono `samples` contain speech, or `None` when the detector
/// is not available and the caller should keep its current behavior.
pub fn contains_speech(samples: &[f32]) -> Option<bool> {
    #[cfg(test)]
    {
        let _ = samples;
        OVERRIDE.with(|verdict| verdict.get())
    }
    #[cfg(not(test))]
    {
        heard(samples).map(|segments| !segments.is_empty())
    }
}

/// First and last speech sample of 16 kHz mono `samples`, or `None` when the
/// detector is not available or hears no speech. The bounds are Silero's own
/// segments, which already carry its `speech_pad_ms`.
pub fn speech_bounds(samples: &[f32]) -> Option<Range<usize>> {
    #[cfg(test)]
    {
        let _ = samples;
        None
    }
    #[cfg(not(test))]
    {
        let segments = heard(samples)?;
        bounds(
            segments
                .iter()
                .map(|segment| (segment.start_sample, segment.end_sample)),
            samples.len(),
        )
    }
}

/// The span of `segments`, each a first speech sample and the first sample
/// after it, in a clip of `len` samples; clamped to the clip.
fn bounds(segments: impl Iterator<Item = (i64, i64)>, len: usize) -> Option<Range<usize>> {
    let (mut first, mut last) = (i64::MAX, i64::MIN);
    let mut heard = false;
    for (start, end) in segments {
        heard = true;
        first = first.min(start);
        last = last.max(end);
    }
    if !heard {
        return None;
    }
    let start = usize::try_from(first.max(0)).unwrap_or(usize::MAX).min(len);
    let end = usize::try_from(last.max(0)).unwrap_or(usize::MAX).min(len);
    (start < end).then_some(start..end)
}

/// `bounds` widened by `margin` samples on each side and clamped to a clip of
/// `len` samples.
pub fn edge_bounds(len: usize, bounds: Range<usize>, margin: usize) -> Range<usize> {
    let start = bounds.start.min(len).saturating_sub(margin);
    let end = bounds.end.min(len).saturating_add(margin).min(len);
    start..end.max(start)
}

/// The detector's segments, or `None` when it is unavailable or failed.
#[cfg(not(test))]
fn heard(samples: &[f32]) -> Option<Vec<VadSegment>> {
    let mut state = lock_state();
    let session = loaded_session(&mut state)?;
    match segments(session, samples) {
        Ok(segments) => Some(segments),
        Err(error) => {
            if !RUN_FAILURE_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(%error, "speech detection failed");
            }
            None
        }
    }
}

/// Loads an installed detector in the background so the first clip does not
/// pay for it. Does nothing without a model path.
pub fn prewarm() {
    #[cfg(not(test))]
    if lock_model().is_some() {
        let _ = std::thread::Builder::new()
            .name("hexasr-speech-detector".into())
            .spawn(|| {
                let _ = loaded_session(&mut lock_state());
            });
    }
}

/// The loaded session, loading it on first use.
#[cfg_attr(test, allow(dead_code))]
fn loaded_session(state: &mut State) -> Option<&mut VadSession> {
    if matches!(state, State::Unloaded) {
        // 锁一次、复用绑定:scrutinee 里的临时守卫活到 match 结束,第二支
        // 再调 lock_model() 会同线程重锁死锁(MODEL 为 None 时必进该支,
        // 2026-11-19 定案的 whisper "死等" 即此处)。
        let model = lock_model();
        *state = match model.as_deref().and_then(|path| load(path).ok()) {
            Some(session) => State::Loaded(session),
            None if model.is_none() => State::Unloaded,
            None => {
                tracing::warn!("could not load the speech detector");
                State::Failed
            }
        };
    }
    match state {
        State::Loaded(session) => Some(session),
        State::Unloaded | State::Failed => None,
    }
}

/// Loads the detector on the CPU: it loads in tens of milliseconds there,
/// while a first GPU start can take seconds and would hold up transcription.
pub fn load(path: &Path) -> Result<VadSession> {
    // Its load would otherwise register every backend unguarded.
    crate::local_transcriber::init_backends(crate::local_transcriber::Device::Cpu)?;
    let model = Model::load_with(
        path,
        &ModelOptions {
            backend: Backend::Cpu,
            ..Default::default()
        },
    )?;
    Ok(model.vad_session()?)
}

/// Whether one run hears speech.
#[cfg(test)]
fn detect(session: &mut VadSession, samples: &[f32]) -> Result<bool> {
    Ok(!segments(session, samples)?.is_empty())
}

fn segments(session: &mut VadSession, samples: &[f32]) -> Result<Vec<VadSegment>> {
    let options = VadOptions {
        threshold: THRESHOLD,
        min_speech_ms: MIN_SPEECH_MS,
        ..VadOptions::default()
    };
    Ok(session.run(samples, &options)?.segments)
}

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Makes `contains_speech` answer `verdict` on this thread until the guard
/// drops. Tests never load the real detector.
#[cfg(test)]
pub(crate) fn override_for_test(verdict: Option<bool>) -> OverrideGuard {
    OverrideGuard(OVERRIDE.with(|current| current.replace(verdict)))
}

#[cfg(test)]
pub(crate) struct OverrideGuard(Option<bool>);

#[cfg(test)]
impl Drop for OverrideGuard {
    fn drop(&mut self) {
        OVERRIDE.with(|current| current.set(self.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_decides_and_restores() {
        {
            let _guard = override_for_test(Some(false));
            assert_eq!(contains_speech(&[0.0; 16_000]), Some(false));
        }
        assert_eq!(contains_speech(&[0.0; 16_000]), None);
    }

    #[test]
    fn detected_segments_bound_the_speech_inside_the_clip() {
        assert_eq!(
            bounds([(1_000, 4_000), (9_000, 12_000)].into_iter(), 16_000),
            Some(1_000..12_000)
        );
        assert_eq!(
            bounds([(12_000, 16_000)].into_iter(), 16_000),
            Some(12_000..16_000)
        );
        assert_eq!(
            bounds([(12_000, 20_000)].into_iter(), 16_000),
            Some(12_000..16_000),
            "a segment past the clip is clamped to it"
        );
        assert_eq!(bounds([].into_iter(), 16_000), None);
        assert_eq!(
            bounds([(16_000, 18_000)].into_iter(), 16_000),
            None,
            "speech past the clip leaves nothing to keep"
        );
    }

    #[test]
    fn a_margin_around_the_speech_clamps_to_the_clip() {
        // 1 s of speech inside 3 s, with the 500 ms margin a Whisper run uses.
        assert_eq!(edge_bounds(48_000, 16_000..32_000, 8_000), 8_000..40_000);
        assert_eq!(
            edge_bounds(48_000, 2_000..4_000, 8_000),
            0..12_000,
            "the leading margin stops at the clip's start"
        );
        assert_eq!(
            edge_bounds(48_000, 44_000..47_000, 8_000),
            36_000..48_000,
            "the trailing margin stops at the clip's end"
        );
        assert_eq!(
            edge_bounds(48_000, 40_000..80_000, 8_000),
            32_000..48_000,
            "bounds past the end still clamp"
        );
    }

    #[test]
    fn a_corrupt_detector_file_is_not_loaded() {
        let path = std::env::temp_dir().join(format!(
            "hexasr-corrupt-speech-detector-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, b"not a gguf file").unwrap();
        let loaded = load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(loaded.is_err());
    }

    #[test]
    #[ignore = "requires HEXASR_VAD_MODEL (Silero VAD GGUF)"]
    fn real_detector_rejects_silence_and_noise() {
        let mut session = load(Path::new(&std::env::var("HEXASR_VAD_MODEL").unwrap())).unwrap();
        assert!(!detect(&mut session, &[0.0; 32_000]).unwrap());
        // -60 dBFS white noise from a fixed linear congruential generator.
        let mut seed = 0x2545_f491_u32;
        let noise: Vec<f32> = (0..32_000)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.001 * 3f32.sqrt()
            })
            .collect();
        assert!(!detect(&mut session, &noise).unwrap());
    }

    fn clip(variable: &str) -> Vec<f32> {
        let mut reader = hound::WavReader::open(std::env::var(variable).unwrap()).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000);
        // hound reads an int wav only as integers (and a float wav only as
        // floats), so scale int samples here.
        match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|sample| sample.unwrap() as f32 / max)
                    .collect()
            }
        }
    }

    #[test]
    #[ignore = "requires HEXASR_VAD_MODEL and HEXASR_VAD_SPEECH_WAV (16 kHz mono float speech)"]
    fn real_detector_hears_speech() {
        let mut session = load(Path::new(&std::env::var("HEXASR_VAD_MODEL").unwrap())).unwrap();
        assert!(detect(&mut session, &clip("HEXASR_VAD_SPEECH_WAV")).unwrap());
    }
}
