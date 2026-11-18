//! Model-file tests, gated on `HEXASR_MODEL` (a path to a GGUF speech
//! model). Every test here skips silently when the variable is unset, so
//! `cargo test` needs no model:
//!
//! ```text
//! HEXASR_MODEL=path/to/model.gguf cargo test -p hex-asr --test model
//! ```
//!
//! The model loads **once per process** and every test shares the same
//! `Arc<Engine>` — five tests are not five 845 MB loads. `HEXASR_LANG`
//! (an ISO code; for models that cannot auto-detect) and `HEXASR_DEVICE`
//! (`auto` the default, `gpu`, or `cpu`) are read at that first load.
//! `auto` prefers a healthy GPU (Vulkan) and falls back to the CPU on its
//! own, so the same suite covers both paths: run it once with no override
//! for the GPU path and once with `HEXASR_DEVICE=cpu` for the CPU path.

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use hex_asr::{Device, Engine, ModelSpec, SessionEvent, SessionEvents, StreamingSession};

fn model_path() -> Option<String> {
    std::env::var("HEXASR_MODEL")
        .ok()
        .filter(|path| !path.is_empty())
}

/// The device the shared engine loads on, from `HEXASR_DEVICE`
/// (`auto`|`gpu`|`cpu`, default `auto`).
fn device_choice() -> Device {
    match std::env::var("HEXASR_DEVICE").ok().as_deref() {
        Some("gpu") => Device::Gpu,
        Some("cpu") => Device::Cpu,
        Some("auto") | None => Device::Auto,
        Some(other) => panic!("HEXASR_DEVICE must be auto, gpu, or cpu, not {other}"),
    }
}

/// The model file loaded once per process, shared by every test here.
/// `None` when `HEXASR_MODEL` is unset (the caller skips).
fn shared_engine() -> Option<Arc<Engine>> {
    static ENGINE: OnceLock<Option<Arc<Engine>>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let path = model_path()?;
            // A parakeet model never advertises auto-detection and needs its
            // en hint plus the 400 ms trailing silence that keeps its closing
            // punctuation (`ModelSpec::unified_english`); other models take
            // the HEXASR_LANG hint or auto-detect.
            let spec = if path.to_ascii_lowercase().contains("parakeet") {
                ModelSpec::unified_english(&path)
            } else {
                match std::env::var("HEXASR_LANG")
                    .ok()
                    .filter(|lang| !lang.is_empty())
                {
                    Some(lang) => ModelSpec {
                        language: Some(lang),
                        ..ModelSpec::new(&path)
                    },
                    None => ModelSpec::new(&path),
                }
            };
            let device = device_choice();
            let started = Instant::now();
            let engine = Arc::new(
                Engine::load(&spec, device)
                    .unwrap_or_else(|error| panic!("could not load {}: {error:#}", path)),
            );
            println!(
                "loaded model={path} device={device:?} load_ms={}",
                started.elapsed().as_millis(),
            );
            Some(engine)
        })
        .clone()
}

/// The loaded engine, or skip the test (no `HEXASR_MODEL`).
macro_rules! engine_or_skip {
    () => {
        match shared_engine() {
            Some(engine) => engine,
            None => return,
        }
    };
}

/// Records everything a session emits, mirroring how a unified-layer sink
/// would behave (this is the whole adapter a wxasr-async backend needs).
#[derive(Default)]
struct Collector {
    events: Vec<SessionEvent>,
}

impl SessionEvents for Collector {
    fn emit(&mut self, event: SessionEvent) -> bool {
        self.events.push(event);
        true
    }
}

/// `gguf_meta` reads only the file header (no weights): the model's own
/// claims about streaming. parakeet advertises `stt.capability.streaming`
/// (its C++ family default never sets it — the KV is the only source);
/// whisper/qwen have no such KV. This test never touches the engine, so
/// it is fast even with `HEXASR_MODEL` set.
#[test]
fn gguf_meta_reads_the_models_own_header() {
    let Some(path) = model_path() else {
        return;
    };
    let meta = hex_asr::gguf_meta(&path)
        .unwrap_or_else(|error| panic!("could not read the header of {path}: {error:#}"));
    println!(
        "arch={} streaming={} name={:?}",
        meta.architecture, meta.streaming, meta.name
    );
    assert!(!meta.architecture.is_empty());
    let lower = path.to_ascii_lowercase();
    if lower.contains("parakeet") {
        assert!(meta.streaming, "parakeet advertises streaming");
    } else if lower.contains("whisper") || lower.contains("qwen") {
        assert!(!meta.streaming, "whisper/qwen have no streaming KV");
    }
}

#[test]
fn a_streaming_session_maps_onto_the_push_contract() {
    let engine = engine_or_skip!();
    let mut session = StreamingSession::open(engine, true);
    let mut out = Collector::default();
    // 1.5 s of silence in arbitrary i16 chunks: partials may come or not
    // (silence), but the clip accumulates whatever the chunk sizes are.
    for _ in 0..15 {
        session.push(&[0_i16; 1600], &mut out);
    }
    assert_eq!(session.pushed_samples(), 24_000);
    // One finish: exactly one Final (or none for empty text), then done.
    session.finish(&mut out);
    session.finish(&mut out);
    let finals = out
        .events
        .iter()
        .filter(|event| matches!(event, SessionEvent::Final(_)))
        .count();
    assert!(
        finals <= 1,
        "a session finalizes at most once: {:?}",
        out.events
    );
    assert_eq!(
        out.events
            .iter()
            .filter(|event| matches!(event, SessionEvent::Partial(_)))
            .count(),
        out.events.len() - finals
    );
}

#[test]
fn loads_and_transcribes_a_clip_offline() {
    let engine = engine_or_skip!();
    let transcribed = engine
        .transcribe_clip(vec![0.0; 24_000])
        .expect("the engine takes the clip")
        .wait()
        .expect("the offline transcription finishes");
    // Silence may transcribe to nothing or to hallucinated text; this only
    // checks the pipeline runs and reports its source.
    assert_eq!(transcribed.source, hex_asr::Source::Offline);
}

#[test]
fn finishes_a_stream_with_the_complete_clip() {
    let engine = engine_or_skip!();
    let mut stream = match engine.begin() {
        Some(stream) => stream,
        None => {
            // A CPU-only device never streams (a stream would re-encode each
            // overlapping window); the offline path covers it.
            return;
        }
    };
    stream.feed(&[0.0; 24_000]);
    let transcribed = engine
        .finish(Some(stream), vec![0.0; 24_000])
        .expect("the engine takes the finish")
        .wait()
        .expect("the finish produces text or falls back to the clip");
    assert!(
        transcribed.source == hex_asr::Source::Stream
            || matches!(transcribed.source, hex_asr::Source::Fallback(_)),
        "a finish comes from the stream or its whole-clip fallback: {:?}",
        transcribed.source
    );
}

#[test]
fn transcribes_with_timings() {
    let engine = engine_or_skip!();
    // Background work yields to a request sent before it; with one shared
    // engine other tests' streams can hold it, so retry until it runs.
    let deadline = Instant::now() + std::time::Duration::from_secs(120);
    let transcript = loop {
        let run = engine
            .background(|model| model.transcribe_timed(vec![0.0; 24_000]))
            .expect("the engine is running");
        match run {
            Some(result) => break result.expect("the timed transcription runs"),
            None if Instant::now() > deadline => {
                panic!("background work stayed preempted for two minutes")
            }
            None => continue,
        }
    };
    assert!(
        transcript.text.len() <= 4_096,
        "a 1.5 s clip cannot run away"
    );
}

/// Reads a wav as 16 kHz mono f32 (any input rate/format hound takes).
fn wav_samples(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|error| panic!("could not open {path}: {error}"));
    let spec = reader.spec();
    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.unwrap() as f32 / max)
                .collect()
        }
    };
    let channels = usize::from(spec.channels);
    if channels > 1 {
        samples = samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
    }
    if spec.sample_rate != hex_asr::SAMPLE_RATE {
        samples = hex_asr::audio::resample(&samples, spec.sample_rate)
            .expect("the wav's sample rate is resampleable");
    }
    samples
}

/// Loads the model, transcribes a short speech wav whole, expects text.
/// `HEXASR_WAV` overrides the clip (default: wxasr's first short TTS
/// sample, whose spoken text is known Chinese).
#[test]
fn transcribes_a_speech_wav_end_to_end() {
    let engine = engine_or_skip!();
    let wav = std::env::var("HEXASR_WAV").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../wxasr/tests/data/tts_short_1.wav"
        )
        .to_string()
    });
    if !std::path::Path::new(&wav).is_file() {
        eprintln!("skipping: {wav} does not exist");
        return;
    }
    let samples = wav_samples(&wav);
    assert!(
        samples.len() > hex_asr::SAMPLE_RATE as usize,
        "the clip must hold real speech"
    );
    let started = Instant::now();
    let transcribed = engine
        .transcribe_clip(samples)
        .expect("the engine takes the clip")
        .wait()
        .expect("the transcription finishes");
    println!(
        "model={} device={} total_ms={} text={:?}",
        std::env::var("HEXASR_MODEL").as_deref().unwrap_or("?"),
        std::env::var("HEXASR_DEVICE").unwrap_or_default(),
        started.elapsed().as_millis(),
        transcribed.text()
    );
    assert!(
        !transcribed.text().trim().is_empty(),
        "a speech clip must transcribe to text"
    );
}
