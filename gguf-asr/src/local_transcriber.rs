//! The GGUF loader: reads the model file where the caller's device choice
//! says, warms it, probes streaming, and recovers onto the CPU when the GPU
//! fails or runs slower.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use eyre::{Context, Result, bail};
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, TimestampKind};

use crate::gguf_session::{OfflineGgufSession, output_truncated, resolve_language};
use crate::gpu_health;
use crate::transcript::{TimestampLevel, Transcript};

/// Where a model loads: the best GPU (`Auto`), a GPU outright, or the CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    /// The best GPU, falling back to (or preferring) the CPU when a GPU
    /// failed or ran slower than the CPU in this process.
    #[default]
    Auto,
    /// The best GPU, trying a failed one again.
    Gpu,
    /// Strict CPU, so no GPU driver loads.
    Cpu,
}

/// A model this library loads: a GGUF file, its language, and the trailing
/// silence its clips pad with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSpec {
    /// The GGUF file to load.
    pub path: PathBuf,
    /// ISO language code to hint, or `None` for auto-detect (a model that
    /// advertises detection).
    pub language: Option<String>,
    /// Silence (samples at 16 kHz) appended after every whole clip, so the
    /// model can emit its final word and closing punctuation. Parakeet
    /// Unified English wants 400 ms; see [`ModelSpec::unified_english`].
    pub trailing_silence_samples: usize,
}

impl ModelSpec {
    /// A model with auto-detected language and no trailing-silence padding.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            language: None,
            trailing_silence_samples: 0,
        }
    }

    /// Parakeet Unified English: English, padded with 400 ms of trailing
    /// silence so its closing punctuation survives (issue 657).
    pub fn unified_english(path: impl Into<PathBuf>) -> Self {
        Self {
            language: Some("en".into()),
            trailing_silence_samples: crate::UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES,
            ..Self::new(path)
        }
    }
}

/// A word or segment's start and end (ms from the clip's start) and text.
pub type TimedSpan = (i64, i64, String);

pub struct LocalTranscriber {
    session: OfflineGgufSession,
    spec: ModelSpec,
    options: RunOptions,
    device_label: String,
    max_audio_samples: Option<usize>,
    stream: Option<transcribe_cpp::StreamOptions>,
    /// The finest timestamps the model gives.
    finest_timestamps: TimestampKind,
    /// The GPU it runs on; `None` on the CPU.
    gpu: Option<transcribe_cpp::Device>,
    /// The device choice it loaded for.
    preference: Device,
    /// A warm run of [`PREWARM_SAMPLES`] once loaded, in milliseconds.
    warm_ms: u128,
}

/// 1.5 s of silence: what loading runs to warm a model and time it.
const PREWARM_SAMPLES: usize = 24_000;

/// An integrated GPU slower than this on a warm 1.5 s run (a fifth of real
/// time) may lose to the CPU, so Auto times the CPU too (#484). Healthy
/// GPUs run it in tens of milliseconds.
const SLOW_WARM_MS: u128 = 300;

/// Registers transcribe.cpp's backends once per process and returns the
/// backend models load on: every GPU, or only the CPU when `device` is
/// `Cpu`, so no GPU driver loads.
pub fn init_backends(device: Device) -> Result<Backend> {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    if let Some(backend) = BACKEND.get() {
        return Ok(*backend);
    }
    transcribe_cpp::init_logging();
    let chosen = if device == Device::Cpu {
        transcribe_cpp::init_backends_with(
            None::<&std::path::Path>,
            transcribe_cpp::BackendMask::for_backend(Backend::Cpu),
        )
        .context("could not initialize the CPU transcription backend")?;
        Backend::Cpu
    } else {
        transcribe_cpp::init_backends_default()
            .context("could not initialize transcription backends")?;
        // Enumerating the devices makes sure every backend started now.
        drop(transcribe_cpp::devices());
        Backend::Auto
    };
    Ok(*BACKEND.get_or_init(|| chosen))
}

/// The compute devices transcribe.cpp knows, one string each.
pub fn devices() -> Result<Vec<String>> {
    init_backends(Device::Auto)?;
    Ok(transcribe_cpp::devices()
        .into_iter()
        .map(|device| {
            format!(
                "{}\t{}\t{} MB\t{}",
                device
                    .index
                    .map_or_else(|| "-".into(), |index| index.to_string()),
                device.kind,
                device.memory_total / (1024 * 1024),
                if device.description.is_empty() {
                    device.name
                } else {
                    device.description
                }
            )
        })
        .collect())
}

impl LocalTranscriber {
    /// Loads and prewarms `spec` where `device` says: on Auto a GPU unless it
    /// failed before in this process ([`gpu_health`], #856), on GPU the best
    /// GPU, on CPU the CPU. A GPU load or prewarm that fails retries on the
    /// CPU, and when the CPU works the GPU is remembered as failed.
    pub fn load(spec: &ModelSpec, device: Device) -> Result<Self> {
        Self::load_where(spec, device)
    }

    fn load_where(spec: &ModelSpec, preference: Device) -> Result<Self> {
        let backend = init_backends(preference)?;
        if backend == Backend::Cpu || preference == Device::Cpu {
            return Self::load_on(spec, Backend::Cpu, None, preference);
        }
        let gpus = gpu_health::gpus();
        let (backend, target) = if preference == Device::Gpu || !gpus.iter().any(gpu_health::failed)
        {
            (Backend::Auto, None)
        } else {
            let healthy = gpus
                .iter()
                .find(|gpu| !gpu_health::failed(gpu) && gpu_health::backend(gpu).is_some());
            match healthy.and_then(|gpu| Some((gpu_health::backend(gpu)?, gpu.clone()))) {
                Some((backend, gpu)) => (backend, Some(gpu)),
                None => {
                    tracing::info!(
                        "this GPU failed or ran slower than the CPU earlier in this process; transcribing on the CPU"
                    );
                    return Self::load_on(spec, Backend::Cpu, None, preference);
                }
            }
        };
        // The GPU this load runs on: the one chosen, else the one Auto picks.
        let expected = target.clone().or_else(|| gpus.into_iter().next());
        match Self::load_on(spec, backend, target, preference) {
            Ok(loaded) => {
                if preference == Device::Gpu
                    && let Some(gpu) = &loaded.gpu
                {
                    gpu_health::forget(gpu);
                }
                if preference == Device::Auto && loaded.slow_igpu() {
                    return Ok(loaded.or_faster_cpu());
                }
                Ok(loaded)
            }
            Err(error) => {
                tracing::warn!(
                    error = format!("{error:#}"),
                    "could not load the transcription model on the GPU; loading it on the CPU"
                );
                let loaded = Self::load_on(spec, Backend::Cpu, None, preference)?;
                // The CPU loads the same model, so the GPU failed, unless it
                // only ran out of memory (another copy can fill it, #1165).
                if let Some(gpu) = &expected {
                    if gpu_health::out_of_memory(&error) {
                        tracing::info!(
                            "the GPU ran out of memory; not remembering it as failed, Auto tries it again next load"
                        );
                    } else {
                        gpu_health::record(gpu, &format!("{error:#}"));
                    }
                }
                Ok(loaded)
            }
        }
    }

    fn load_on(
        spec: &ModelSpec,
        backend: Backend,
        target: Option<transcribe_cpp::Device>,
        preference: Device,
    ) -> Result<Self> {
        let path = &spec.path;
        if !path.is_file() {
            bail!("the transcription model {} does not exist", path.display());
        }
        let model = Model::load_with(
            path,
            &ModelOptions {
                backend,
                device: target,
                ..Default::default()
            },
        )
        .with_context(|| format!("could not load transcription model from {}", path.display()))?;
        let device = model.device()?;
        let variant = model.variant();
        let architecture = model.arch();
        let capabilities = model.capabilities()?;
        let language = resolve_language(path, spec.language.as_deref(), &capabilities)?;
        let options = RunOptions {
            timestamps: if capabilities.max_timestamp_kind == TimestampKind::None {
                TimestampKind::None
            } else {
                TimestampKind::Segment
            },
            language,
            ..Default::default()
        };
        let device_label = format!("{} ({})", device.description, device.kind);
        tracing::info!(
            backend = device.kind,
            device = device.description,
            key = gpu_health::device_key(&device),
            variant,
            "loaded local transcription model"
        );
        let session = OfflineGgufSession::new(model)?;
        let mut transcriber = Self {
            session,
            spec: spec.clone(),
            options,
            device_label,
            max_audio_samples: max_audio_chunk_samples(&architecture, capabilities.max_audio_ms),
            stream: None,
            finest_timestamps: match capabilities.max_timestamp_kind {
                TimestampKind::Word | TimestampKind::Token => TimestampKind::Word,
                kind => kind,
            },
            gpu: matches!(
                device.device_type,
                transcribe_cpp::DeviceType::Gpu | transcribe_cpp::DeviceType::Igpu
            )
            .then(|| device.clone()),
            preference,
            warm_ms: 0,
        };
        let started = Instant::now();
        transcriber.warm_on_silence()?;
        tracing::info!(
            prewarm_ms = started.elapsed().as_millis(),
            "prewarmed local transcription model"
        );
        transcriber.stream = transcriber
            .session
            .probe_stream(&transcriber.options, &architecture);
        // Warm now, so this times the device rather than its first run's
        // shader compiles.
        let started = Instant::now();
        transcriber.warm_on_silence()?;
        transcriber.warm_ms = started.elapsed().as_millis();
        Ok(transcriber)
    }

    /// Runs [`PREWARM_SAMPLES`] of silence through the model. A decode that
    /// still runs to the output limit ran the whole graph on the device, so
    /// that is a warm model, not a failed device.
    fn warm_on_silence(&mut self) -> Result<()> {
        let run = |this: &mut Self| {
            this.session
                .warm(&vec![0.0; PREWARM_SAMPLES], &this.options)
                .map(drop)
                .context("transcription failed")
        };
        let result = match run(self) {
            // A GPU that fails mid-run reloads on the CPU, as a transcript
            // does; the CPU model is warmed by its own load.
            Err(error) if self.gpu_failed(&error) => self.recover_on_cpu(&error),
            result => result,
        };
        match result {
            Err(error) if output_truncated(&error) => {
                tracing::info!(
                    error = format!("{error:#}"),
                    "the warm-up run on silence hit the output limit; the model is warm"
                );
                Ok(())
            }
            result => result,
        }
    }

    pub fn stream_options(&self) -> Option<&transcribe_cpp::StreamOptions> {
        self.stream.as_ref()
    }

    /// The finest timestamp kind this model can answer at (its capability,
    /// as runs ask for it); `None` means text-only transcripts, with no rows
    /// to merge diarize turns into.
    pub fn finest_timestamps(&self) -> transcribe_cpp::TimestampKind {
        self.finest_timestamps
    }

    /// Silence a live stream of this model hears before it finalizes, the
    /// same trailing silence its whole-clip runs get.
    pub fn stream_trailing_silence(&self) -> usize {
        self.spec.trailing_silence_samples
    }

    /// The trailing silence this model's whole clips pad with.
    pub fn trailing_silence_samples(&self) -> usize {
        self.spec.trailing_silence_samples
    }

    /// The spec this transcriber was loaded for.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn with_stream<R>(
        &mut self,
        stream_options: &transcribe_cpp::StreamOptions,
        cancel: &transcribe_cpp::CancelToken,
        drive: impl FnOnce(&mut transcribe_cpp::Stream<'_>) -> R,
    ) -> transcribe_cpp::Result<R> {
        self.session
            .with_stream(&self.options, stream_options, cancel, drive)
    }

    /// Brings an idle model's weights back before audio arrives, as loading
    /// does; a model that ran recently is already warm.
    pub fn prewarm(&mut self) -> Result<()> {
        if self.session.ran_recently() {
            return Ok(());
        }
        self.warm_on_silence()
            .context("could not prewarm transcription model")
    }

    /// Transcribes a complete 16 kHz clip, padded as a dictation's clip is:
    /// the minimum duration, then the model's trailing silence.
    pub fn transcribe_clip(&mut self, mut samples: Vec<f32>) -> Result<Transcript> {
        crate::pad_gguf_samples(&mut samples, self.spec.trailing_silence_samples);
        self.transcribe(&samples)
    }

    /// [`Self::transcribe_clip`] with the model's own segment and word times,
    /// chunked and padded exactly as the untimed run chunks and pads it, so
    /// the text matches and each chunk's times sit on the clip's timeline.
    pub fn transcribe_timed(&mut self, mut samples: Vec<f32>) -> Result<Transcript> {
        crate::pad_gguf_samples(&mut samples, self.spec.trailing_silence_samples);
        let options = RunOptions {
            timestamps: self.finest_timestamps,
            ..self.options.clone()
        };
        match self
            .session
            .transcribe_chunked_timed(&samples, &options, self.max_audio_samples)
        {
            Err(error) if self.gpu_failed(&error) => {
                self.recover_on_cpu(&error)?;
                self.session
                    .transcribe_chunked_timed(&samples, &options, self.max_audio_samples)
            }
            result => result,
        }
    }

    /// The finest native timings this model gives, as a timed request needs
    /// them. Never the level a caller asked for.
    pub fn timestamps(&self) -> TimestampLevel {
        crate::gguf_session::timestamp_level(self.finest_timestamps)
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    /// Text and word times (ms from the clip's start), in chunks the model
    /// takes whole. A model without word times gives its segments instead.
    pub fn transcribe_words(&mut self, samples: &[f32]) -> Result<(String, Vec<TimedSpan>)> {
        let transcript = self.timed(samples)?;
        let spans = if transcript.words.is_empty() {
            transcript.segments
        } else {
            transcript.words
        };
        let words = spans
            .into_iter()
            .map(|word| (word.start_ms, word.end_ms, word.text))
            .collect();
        Ok((transcript.text, words))
    }

    /// One chunk with segment and word timings, padded as a clip is.
    pub fn transcribe_segments(&mut self, mut samples: Vec<f32>) -> Result<Transcript> {
        crate::audio::pad_minimum(&mut samples);
        self.timed(&samples)
    }

    /// Text, segments, and words, in chunks the model takes whole.
    fn timed(&mut self, samples: &[f32]) -> Result<Transcript> {
        match self.timed_here(samples) {
            Err(error) if self.gpu_failed(&error) => {
                self.recover_on_cpu(&error)?;
                self.timed_here(samples)
            }
            result => result,
        }
    }

    /// Whether this model runs on an integrated GPU slow enough that the
    /// CPU may be faster.
    fn slow_igpu(&self) -> bool {
        self.warm_ms > SLOW_WARM_MS
            && self
                .gpu
                .as_ref()
                .is_some_and(|gpu| gpu.device_type == transcribe_cpp::DeviceType::Igpu)
    }

    /// This model, or the same one on the CPU when the CPU runs it faster;
    /// then the GPU is remembered as slow, so later loads go straight to the
    /// CPU (#484).
    fn or_faster_cpu(self) -> Self {
        let cpu = match Self::load_on(&self.spec, Backend::Cpu, None, self.preference) {
            Ok(cpu) => cpu,
            Err(error) => {
                tracing::warn!(
                    error = format!("{error:#}"),
                    "could not time the model on the CPU; keeping the GPU"
                );
                return self;
            }
        };
        tracing::info!(
            gpu_ms = self.warm_ms,
            cpu_ms = cpu.warm_ms,
            "timed a warm 1.5 s run on the integrated GPU and the CPU"
        );
        if cpu.warm_ms >= self.warm_ms {
            return self;
        }
        if let Some(gpu) = &self.gpu {
            gpu_health::record_slow(
                gpu,
                &format!(
                    "a warm 1.5 s run took {} ms on the GPU and {} ms on the CPU",
                    self.warm_ms, cpu.warm_ms
                ),
            );
        }
        cpu
    }

    /// Whether `error` came from the GPU this model runs on failing.
    fn gpu_failed(&self, error: &eyre::Report) -> bool {
        self.gpu.is_some() && gpu_health::device_error(error)
    }

    /// Reloads this model on the CPU after its GPU failed mid-run (a hang
    /// or reset comes back as an error, #856), remembering the GPU once the
    /// CPU works.
    fn recover_on_cpu(&mut self, error: &eyre::Report) -> Result<()> {
        tracing::warn!(
            error = format!("{error:#}"),
            "the GPU failed while transcribing; reloading the model on the CPU"
        );
        let gpu = self.gpu.take();
        let cpu = Self::load_on(&self.spec, Backend::Cpu, None, self.preference)?;
        if let Some(gpu) = &gpu
            && !gpu_health::out_of_memory(error)
        {
            gpu_health::record(gpu, &format!("{error:#}"));
        }
        *self = cpu;
        Ok(())
    }

    fn timed_here(&mut self, samples: &[f32]) -> Result<Transcript> {
        use crate::gguf_session::{chunk_ranges, sample_offset_ms};
        const CHUNK: usize = 30 * 16_000;
        let chunk = self.max_audio_samples.map_or(CHUNK, |max| max.min(CHUNK));
        let options = RunOptions {
            timestamps: self.finest_timestamps,
            ..self.options.clone()
        };
        let span = |offset: i64, t0: i64, t1: i64, text: String, speaker: Option<u32>| {
            let text = text.trim().to_owned();
            (!text.is_empty()).then(|| crate::transcript::TranscriptSegment {
                start_ms: t0.max(0) + offset,
                end_ms: t1.max(t0) + offset,
                text,
                speaker,
            })
        };
        let speaker =
            |id: i32| (id > 0).then_some(u32::try_from(id).expect("speaker ids are small"));
        let mut timed = Transcript::default();
        let mut texts = Vec::new();
        for range in chunk_ranges(samples, chunk) {
            let offset = sample_offset_ms(range.start);
            let transcript = self
                .session
                .run(&samples[range], &options)
                .context("transcription failed")?;
            let text = transcript.text.trim();
            if !text.is_empty() {
                texts.push(text.to_owned());
            }
            if timed.language.is_none() {
                timed.language = transcript.language;
            }
            timed.segments.extend(
                transcript
                    .segments
                    .into_iter()
                    .filter_map(|s| span(offset, s.t0_ms, s.t1_ms, s.text, speaker(s.speaker_id))),
            );
            timed.words.extend(
                transcript
                    .words
                    .into_iter()
                    .filter_map(|w| span(offset, w.t0_ms, w.t1_ms, w.text, None)),
            );
        }
        timed.text = texts.join(" ");
        Ok(timed)
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript> {
        let run = |this: &mut Self| {
            this.session
                .transcribe_chunked(samples, &this.options, this.max_audio_samples)
        };
        match run(self) {
            Err(error) if self.gpu_failed(&error) => {
                self.recover_on_cpu(&error)?;
                run(self)
            }
            result => result,
        }
    }
}

/// The most audio one run takes, from the model's own advertised limit:
/// Cohere's reference processor chunks long audio at 35 seconds, and Whisper
/// decodes one 30-second window at a time (handed a longer clip its own
/// long-form loop fills a near-silent remainder with invented text, #1770).
fn max_audio_chunk_samples(architecture: &str, max_audio_ms: i64) -> Option<usize> {
    let maximum_ms = match (architecture, max_audio_ms) {
        ("cohere_asr", limit) if limit > 0 => limit.min(35_000),
        ("cohere_asr", _) => 35_000,
        ("whisper", limit) if limit > 0 => limit.min(30_000),
        ("whisper", _) => 30_000,
        (_, limit) => limit,
    };
    usize::try_from(maximum_ms)
        .ok()
        .filter(|milliseconds| *milliseconds > 0)?
        .checked_mul(16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_limits_come_from_the_architecture_and_the_advertised_maximum() {
        assert_eq!(max_audio_chunk_samples("cohere_asr", 0), Some(35_000 * 16));
        assert_eq!(
            max_audio_chunk_samples("cohere_asr", 120_000),
            Some(35_000 * 16),
            "the reference processor's window caps the encoder's"
        );
        assert_eq!(max_audio_chunk_samples("whisper", 0), Some(30_000 * 16));
        assert_eq!(
            max_audio_chunk_samples("parakeet", 0),
            None,
            "no advertised limit means the model takes the clip whole"
        );
        assert_eq!(max_audio_chunk_samples("parakeet", 10_000), Some(160_000));
    }

    #[test]
    fn a_missing_model_file_fails_before_any_backend_work() {
        let spec = ModelSpec::new("no-such-model.gguf");
        let error = match LocalTranscriber::load_on(&spec, Backend::Cpu, None, Device::Cpu) {
            Ok(_) => panic!("a missing model file must fail"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("does not exist"), "{error}");
    }

    #[test]
    fn unified_english_pads_its_clips() {
        let spec = ModelSpec::unified_english("model.gguf");
        assert_eq!(spec.language.as_deref(), Some("en"));
        assert_eq!(
            spec.trailing_silence_samples,
            crate::UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES
        );
        let plain = ModelSpec::new("model.gguf");
        assert_eq!(plain.language, None);
        assert_eq!(plain.trailing_silence_samples, 0);
    }
}
