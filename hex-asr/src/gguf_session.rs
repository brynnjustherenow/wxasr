//! The `transcribe-cpp` session owner. It recreates the native session
//! around every run (and every live stream) so retained scheduler scratch
//! never outlives the inference that sized it.
//!
//! The fork of transcribe.cpp Hex uses adds Parakeet vocabulary biasing and a
//! context-prompt vocabulary slot; this build stands on the crates.io
//! release, so no vocabulary is fed to any run.

use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::Range;
use std::time::{Duration, Instant};

use eyre::{Context as _, Result, bail};
use transcribe_cpp::Capabilities;

use crate::transcript::{TimestampLevel, Transcript, TranscriptSegment};

/// 8 s of audio reaches every stream window shape a dictation can use (the
/// left context grows to 5.6 s), so the GPU driver compiles those kernels
/// during launch.
const STREAM_PREWARM_SAMPLES: usize = 8 * 16_000;

thread_local! {
    static JOB_CANCEL: RefCell<Option<transcribe_cpp::CancelToken>> = const { RefCell::new(None) };
}

/// Runs `work` with `cancel` installed on every offline run it makes on this
/// thread, so cancelling one engine job aborts its model between decode steps
/// and between chunks. The engine's offline closures reach the session through
/// model types that know nothing of jobs.
pub fn with_job_cancel<R>(cancel: &transcribe_cpp::CancelToken, work: impl FnOnce() -> R) -> R {
    struct Restore(Option<transcribe_cpp::CancelToken>);
    impl Drop for Restore {
        fn drop(&mut self) {
            JOB_CANCEL.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(JOB_CANCEL.with(|slot| slot.borrow_mut().replace(cancel.clone())));
    work()
}

pub fn job_cancel() -> Option<transcribe_cpp::CancelToken> {
    JOB_CANCEL.with(|slot| slot.borrow().clone())
}

/// The language a run hints at: the caller's explicit choice, or `None` for
/// auto-detect, which only a model advertising detection can take.
pub fn resolve_language(
    path: &std::path::Path,
    language: Option<&str>,
    capabilities: &Capabilities,
) -> Result<Option<String>> {
    let language = match language {
        Some(language) if language != "auto" => Some(language.to_string()),
        _ => None,
    };
    if language.is_none() && !capabilities.supports_language_detect {
        bail!(
            "{} does not advertise automatic language detection; pass an explicit language",
            path.display()
        );
    }
    Ok(language)
}

// On GPU devices only the per-token decoder runs on CPU; extra spin-waiting ggml
// workers (default up to 8) stall it under load, so keep it on the calling thread.
fn session_options(device_type: transcribe_cpp::DeviceType) -> transcribe_cpp::SessionOptions {
    match device_type {
        transcribe_cpp::DeviceType::Gpu | transcribe_cpp::DeviceType::Igpu => {
            transcribe_cpp::SessionOptions {
                n_threads: 1,
                ..Default::default()
            }
        }
        _ => transcribe_cpp::SessionOptions::default(),
    }
}

/// Why one whole-clip run happens.
#[derive(Clone, Copy)]
enum RunKind {
    /// A transcript for a caller.
    Transcript,
    /// A warm-up: the weights come back onto the device before audio arrives.
    Warmup,
}

impl RunKind {
    /// Whether a decode that stopped at its output limit still counts as a
    /// run. A warm-up ran the whole graph on the device either way, so a
    /// follow-up warm-up need not repeat it; a failed transcript does not
    /// count.
    fn counts_when_truncated(self) -> bool {
        matches!(self, Self::Warmup)
    }
}

/// A run this recent leaves the weights resident (ggml keeps them so for three
/// minutes), so a prewarm after it would only repeat work.
const RECENT_RUN: Duration = Duration::from_secs(60);

pub struct OfflineGgufSession {
    model: transcribe_cpp::Model,
    session: Option<transcribe_cpp::Session>,
    options: transcribe_cpp::SessionOptions,
    /// When the last run or live stream completed; `None` until the first.
    last_run: Option<Instant>,
    /// The file's architecture, read once: Whisper trims its near-silent
    /// edges before a run.
    architecture: String,
}

impl OfflineGgufSession {
    pub fn new(model: transcribe_cpp::Model) -> transcribe_cpp::Result<Self> {
        let options = session_options(model.device()?.device_type);
        let session = model.session_with(&options)?;
        let architecture = model.arch();
        Ok(Self {
            model,
            session: Some(session),
            options,
            last_run: None,
            architecture,
        })
    }

    /// Whether a run, live stream, or warm-up completed within
    /// [`RECENT_RUN`]. A warm-up that stopped at its output limit counts: it
    /// decoded on the device.
    pub fn ran_recently(&self) -> bool {
        ran_within(self.last_run, Instant::now(), RECENT_RUN)
    }

    /// The model's architecture, read once at load.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Streams only on a GPU (Metal, Vulkan): a CPU stream re-encodes each
    /// overlapping window and would cost several times the offline compute.
    /// The probe also prewarms the stream graph (see
    /// [`STREAM_PREWARM_SAMPLES`]). `HEXASR_STREAMING=0` keeps the model
    /// offline and `HEXASR_STREAMING=1` streams on any device (diagnostics
    /// and CI); `HEXASR_STREAM_CHUNK_MS` overrides the chunk and right
    /// context.
    pub fn probe_stream(
        &mut self,
        options: &transcribe_cpp::RunOptions,
        architecture: &str,
    ) -> Option<transcribe_cpp::StreamOptions> {
        use transcribe_cpp::DeviceType;
        let forced = match std::env::var("HEXASR_STREAMING").as_deref() {
            Ok("0") => return None,
            Ok("1") => true,
            _ => false,
        };
        if !self
            .model
            .capabilities()
            .is_ok_and(|capabilities| capabilities.supports_streaming)
        {
            return None;
        }
        let device_type = self.model.device().ok()?.device_type;
        if !forced && !matches!(device_type, DeviceType::Gpu | DeviceType::Igpu) {
            return None;
        }
        let chunk_ms = std::env::var("HEXASR_STREAM_CHUNK_MS")
            .ok()
            .and_then(|value| value.parse::<i32>().ok());
        let stream_options = transcribe_cpp::StreamOptions {
            family: (architecture == "parakeet" && chunk_ms.is_some()).then_some(
                transcribe_cpp::StreamExtension::ParakeetBuffered(
                    transcribe_cpp::ParakeetBufferedStreamOptions {
                        left_ms: None,
                        chunk_ms,
                        right_ms: chunk_ms,
                    },
                ),
            ),
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let probe = self.with_stream(
            options,
            &stream_options,
            &transcribe_cpp::CancelToken::new(),
            |stream| {
                stream.feed(&vec![0.0; STREAM_PREWARM_SAMPLES])?;
                stream.finalize().map(|_| ())
            },
        );
        match probe {
            Ok(Ok(())) => {
                tracing::info!(
                    ?device_type,
                    chunk_ms,
                    prewarm_ms = started.elapsed().as_millis(),
                    "streaming transcription is available"
                );
                Some(stream_options)
            }
            Ok(Err(error)) | Err(error) => {
                tracing::warn!(%error, "model advertises streaming but could not stream");
                None
            }
        }
    }

    /// Runs `drive` against one live stream on this session, then rotates the
    /// session like [`Self::run`] so the stream's buffered audio and scratch
    /// are released. `cancel` stays installed only for the stream's lifetime.
    pub fn with_stream<R>(
        &mut self,
        options: &transcribe_cpp::RunOptions,
        stream_options: &transcribe_cpp::StreamOptions,
        cancel: &transcribe_cpp::CancelToken,
        drive: impl FnOnce(&mut transcribe_cpp::Stream<'_>) -> R,
    ) -> transcribe_cpp::Result<R> {
        let mut session = match self.session.take() {
            Some(session) => session,
            None => self.model.session_with(&self.options)?,
        };
        session.set_cancel_token(cancel);
        let result = session
            .stream(options, stream_options)
            .map(|mut stream| drive(&mut stream));
        drop(session);
        if result.is_ok() {
            self.last_run = Some(Instant::now());
        }
        self.session = self.model.session_with(&self.options).ok();
        result
    }

    pub fn run(
        &mut self,
        samples: &[f32],
        options: &transcribe_cpp::RunOptions,
    ) -> transcribe_cpp::Result<transcribe_cpp::Transcript> {
        self.run_as(samples, options, RunKind::Transcript)
    }

    /// Runs `samples` exactly like [`Self::run`], for a warm-up. A decode that
    /// stops at the output limit leaves the model warm and counts as a run,
    /// so a prewarm right after the load is not repeated.
    pub fn warm(
        &mut self,
        samples: &[f32],
        options: &transcribe_cpp::RunOptions,
    ) -> transcribe_cpp::Result<transcribe_cpp::Transcript> {
        self.run_as(samples, options, RunKind::Warmup)
    }

    fn run_as(
        &mut self,
        samples: &[f32],
        options: &transcribe_cpp::RunOptions,
        kind: RunKind,
    ) -> transcribe_cpp::Result<transcribe_cpp::Transcript> {
        // transcribe-cpp retains input-sized scheduler buffers for the
        // session lifetime. Drop each used session before creating its
        // successor so a long inference cannot pin that high-water mark or
        // overlap two sessions' persistent decoder state.
        let mut session = match self.session.take() {
            Some(session) => session,
            None => self.model.session_with(&self.options)?,
        };
        if let Some(cancel) = job_cancel() {
            session.set_cancel_token(&cancel);
        }
        let result =
            keep_repetition_partial(session.run(samples, options)).map(|mut transcript| {
                drop_subtitle_credit(&mut transcript);
                transcript
            });
        drop(session);
        if result.is_ok()
            || (kind.counts_when_truncated()
                && matches!(&result, Err(transcribe_cpp::Error::OutputTruncated { .. })))
        {
            self.last_run = Some(Instant::now());
        }
        match self.model.session_with(&self.options) {
            Ok(session) => {
                self.session = Some(session);
                result
            }
            Err(recovery_error) => match result {
                Ok(_) => Err(recovery_error),
                Err(run_error) => Err(run_error),
            },
        }
    }

    /// The audio a whole-clip run gets, and where it starts in `samples`.
    /// Whisper decodes a whole 30 s window and fills a near-silent opening or
    /// tail with text nobody said (a Turkish "thanks for watching", #1770), so
    /// its clip keeps only the speech the detector hears plus
    /// [`WHISPER_EDGE_MARGIN_SAMPLES`]. Another model, a detector that is
    /// unavailable, and a clip it hears no speech in all keep every sample.
    fn whisper_audio<'a>(&self, samples: &'a [f32]) -> (Cow<'a, [f32]>, usize) {
        let bounds = if self.architecture == "whisper" {
            crate::speech_detector::speech_bounds(samples)
        } else {
            None
        };
        let (audio, trimmed) = trimmed_audio(samples, &self.architecture, bounds);
        if let Cow::Owned(audio) = &audio {
            tracing::info!(
                trimmed_ms = trimmed / 16,
                kept_ms = audio.len() / 16,
                "trimmed near-silent edges before a Whisper run"
            );
        }
        (audio, trimmed)
    }

    pub fn transcribe_chunked(
        &mut self,
        samples: &[f32],
        options: &transcribe_cpp::RunOptions,
        max_audio_samples: Option<usize>,
    ) -> Result<String> {
        let (audio, _) = self.whisper_audio(samples);
        let samples = audio.as_ref();
        let Some(max_samples) =
            max_audio_samples.filter(|&max_samples| samples.len() > max_samples)
        else {
            return self
                .run(samples, options)
                .map(|transcript| transcript.text)
                .context("transcription failed");
        };
        let mut output = Vec::new();
        for range in chunk_ranges(samples, max_samples) {
            let samples = &samples[range];
            if job_cancel().is_some_and(|cancel| cancel.is_cancelled()) {
                bail!("transcription was cancelled");
            }
            if !chunk_may_hold_speech(samples) {
                tracing::info!(
                    chunk_ms = samples.len() / 16,
                    "skipped a chunk of a long clip with no speech"
                );
                continue;
            }
            let mut chunk = samples.to_vec();
            crate::audio::pad_minimum(&mut chunk);
            let transcript = self.run(&chunk, options).context("transcription failed")?;
            let text = transcript.text.trim();
            if !text.is_empty() {
                output.push(text.to_string());
            }
        }
        Ok(output.join(" "))
    }

    /// [`Self::transcribe_chunked`] with the model's own segment and word
    /// times, moved onto the clip's timeline by each chunk's sample position.
    /// Text is joined exactly as the untimed run joins it.
    pub fn transcribe_chunked_timed(
        &mut self,
        samples: &[f32],
        options: &transcribe_cpp::RunOptions,
        max_audio_samples: Option<usize>,
    ) -> Result<Transcript> {
        let (audio, trimmed) = self.whisper_audio(samples);
        let samples = audio.as_ref();
        let Some(max_samples) =
            max_audio_samples.filter(|&max_samples| samples.len() > max_samples)
        else {
            let transcript = self.run(samples, options).context("transcription failed")?;
            return Ok(timed(transcript, chunk_offset_ms(trimmed, 0)));
        };
        let mut timed = Transcript::default();
        let mut texts = Vec::new();
        for range in chunk_ranges(samples, max_samples) {
            let offset = chunk_offset_ms(trimmed, range.start);
            let samples = &samples[range];
            if job_cancel().is_some_and(|cancel| cancel.is_cancelled()) {
                bail!("transcription was cancelled");
            }
            if !chunk_may_hold_speech(samples) {
                tracing::info!(
                    chunk_ms = samples.len() / 16,
                    "skipped a chunk of a long clip with no speech"
                );
                continue;
            }
            let mut chunk = samples.to_vec();
            crate::audio::pad_minimum(&mut chunk);
            let transcript = self.run(&chunk, options).context("transcription failed")?;
            let text = transcript.text.trim();
            if !text.is_empty() {
                texts.push(text.to_string());
            }
            // The offset comes from where the range sits in the clip, never
            // from how many chunks produced text.
            let spans = timed_spans(transcript, offset);
            timed.segments.extend(spans.segments);
            timed.words.extend(spans.words);
        }
        timed.text = texts.join(" ");
        Ok(timed)
    }
}

/// A whole clip's segments and words, from its single run.
fn timed(transcript: transcribe_cpp::Transcript, offset_ms: i64) -> Transcript {
    let text = transcript.text.clone();
    let spans = timed_spans(transcript, offset_ms);
    Transcript {
        text,
        segments: spans.segments,
        words: spans.words,
    }
}

/// The native timing level a run options timestamp kind asks for.
pub fn timestamp_level(kind: transcribe_cpp::TimestampKind) -> TimestampLevel {
    match kind {
        transcribe_cpp::TimestampKind::Word | transcribe_cpp::TimestampKind::Token => {
            TimestampLevel::Word
        }
        transcribe_cpp::TimestampKind::Segment | transcribe_cpp::TimestampKind::Auto => {
            TimestampLevel::Segment
        }
        transcribe_cpp::TimestampKind::None => TimestampLevel::None,
    }
}

/// 100 ms at 16 kHz: the span a long clip is cut through.
const QUIET_SPAN_SAMPLES: usize = 1_600;
/// How far back from a window's end the quietest span is looked for.
const QUIET_SEARCH_SAMPLES: usize = 5 * 16_000;

/// Where a window is cut into the next one: the middle of the quietest 100 ms
/// in its last five seconds (or its last half, when that is shorter), never
/// its first sample. `samples` must hold at least `max_samples`, and only
/// those are read, so a streaming reader can cut before it has read the rest
/// of the clip.
pub fn quiet_cut(samples: &[f32], max_samples: usize) -> usize {
    let max_samples = max_samples.max(1);
    let search = QUIET_SEARCH_SAMPLES.min(max_samples / 2).max(1);
    let span = QUIET_SPAN_SAMPLES.min(search);
    let quiet =
        max_samples - search + quietest_span(&samples[max_samples - search..max_samples], span);
    quiet.max(1)
}

/// Splits a clip into consecutive ranges of at most `max_samples`, cutting
/// each window at the middle of the quietest 100 ms in its last five seconds
/// (or half the window, when that is shorter), adapted from Cohere's reference
/// processor. An exact sample cut lands mid-word, and each slice starts a
/// fresh sentence with a capital and a closing full stop, so a long dictation
/// reads as many sentences instead of one. Each cut is [`quiet_cut`]'s.
///
/// Every sample is covered exactly once by a non-empty range. Ties pick the
/// latest span, so the ranges are deterministic. The search is energy-based,
/// so continuous speech with no quieter stretch there still cuts at its
/// quietest point, which can land inside a word.
pub fn chunk_ranges(samples: &[f32], max_samples: usize) -> Vec<Range<usize>> {
    let max_samples = max_samples.max(1);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        if samples.len() - start <= max_samples {
            ranges.push(start..samples.len());
            break;
        }
        let cut = start + quiet_cut(&samples[start..], max_samples);
        ranges.push(start..cut);
        start = cut;
    }
    ranges
}

/// The midpoint of the quietest `span` samples, the latest one on a tie.
fn quietest_span(samples: &[f32], span: usize) -> usize {
    if samples.is_empty() {
        return 0;
    }
    let span = span.clamp(1, samples.len());

    let sample_energy = |sample: f32| f64::from(sample) * f64::from(sample);
    let mut energy = samples[..span]
        .iter()
        .copied()
        .map(sample_energy)
        .sum::<f64>();
    let mut quietest = energy;
    let mut quietest_start = 0;
    for next in 1..=samples.len() - span {
        let leaving = sample_energy(samples[next - 1]);
        let entering = sample_energy(samples[next + span - 1]);
        energy += entering - leaving;
        // Ties keep the latest span.
        if energy <= quietest {
            quietest = energy;
            quietest_start = next;
        }
    }
    quietest_start + span / 2
}

/// Where `sample` sits in the clip, in whole milliseconds.
pub fn sample_offset_ms(sample: usize) -> i64 {
    i64::try_from(sample / 16).unwrap_or(i64::MAX)
}

/// Where a chunk starting at `chunk_start` in the trimmed audio sits in the
/// original clip, in whole milliseconds. A trimmed clip's segment and word
/// times move by this, so they stay on the clip's timeline; the whole clip is
/// the chunk that starts at 0.
fn chunk_offset_ms(trim_start: usize, chunk_start: usize) -> i64 {
    sample_offset_ms(trim_start.saturating_add(chunk_start))
}

fn timed_spans(transcript: transcribe_cpp::Transcript, offset_ms: i64) -> Transcript {
    let span = |start_ms: i64, end_ms: i64, text: String| {
        let text = text.trim().to_owned();
        let start_ms = start_ms.max(0);
        (!text.is_empty()).then(|| TranscriptSegment {
            start_ms: start_ms.saturating_add(offset_ms),
            end_ms: end_ms.max(start_ms).saturating_add(offset_ms),
            text,
        })
    };
    Transcript {
        segments: transcript
            .segments
            .into_iter()
            .filter_map(|segment| span(segment.t0_ms, segment.t1_ms, segment.text))
            .collect(),
        words: transcript
            .words
            .into_iter()
            .filter_map(|word| span(word.t0_ms, word.t1_ms, word.text))
            .collect(),
        ..Default::default()
    }
}

/// How much audio a trimmed Whisper run keeps around the speech the detector
/// hears, on each side.
const WHISPER_EDGE_MARGIN_SAMPLES: usize = 8_000;

/// The span of a clip of `len` samples a whole-clip run keeps. A Whisper clip
/// loses everything outside its detected speech plus
/// [`WHISPER_EDGE_MARGIN_SAMPLES`]; any other model, and a clip without speech
/// bounds, keeps every sample.
fn whisper_span(len: usize, architecture: &str, bounds: Option<Range<usize>>) -> Range<usize> {
    match (architecture, bounds) {
        ("whisper", Some(bounds)) => {
            crate::speech_detector::edge_bounds(len, bounds, WHISPER_EDGE_MARGIN_SAMPLES)
        }
        _ => 0..len,
    }
}

/// `samples` for a whole-clip run: the span [`whisper_span`] keeps, padded to
/// the minimum duration every chunk and every caller pads to, and where that
/// span starts in `samples`. A clip the span covers whole is borrowed, never
/// copied or padded.
fn trimmed_audio<'a>(
    samples: &'a [f32],
    architecture: &str,
    bounds: Option<Range<usize>>,
) -> (Cow<'a, [f32]>, usize) {
    let span = whisper_span(samples.len(), architecture, bounds);
    if span.start == 0 && span.end == samples.len() {
        return (Cow::Borrowed(samples), 0);
    }
    let mut audio = samples[span.clone()].to_vec();
    crate::audio::pad_minimum(&mut audio);
    (Cow::Owned(audio), span.start)
}

/// Whether one chunk of a long clip goes to the model. The recording as a
/// whole held speech, but a chunk of it can be only silence (a locked
/// dictation left waiting before the words, or a pause between two pieces),
/// and models such as Cohere and Whisper invent long, repeating text for that
/// (#1194, #1770), or run to their generation cap (#968). Without the
/// detector every chunk runs.
pub fn chunk_may_hold_speech(samples: &[f32]) -> bool {
    crate::speech_detector::contains_speech(samples) != Some(false)
}

/// Whether `error` is a decode stopped at its output limit. A decoder that
/// runs to its generation cap on silence still ran the whole graph on the
/// device, so a warm-up that ends this way leaves the model warm: it is not a
/// failed device, and not a failed model.
pub fn output_truncated(error: &eyre::Report) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<transcribe_cpp::Error>(),
            Some(transcribe_cpp::Error::OutputTruncated { .. })
        )
    })
}

/// transcribe-cpp stops a decoder that starts looping and reports the text
/// before the loop as an error; keep returning that text.
fn keep_repetition_partial(
    result: transcribe_cpp::Result<transcribe_cpp::Transcript>,
) -> transcribe_cpp::Result<transcribe_cpp::Transcript> {
    match result {
        Err(transcribe_cpp::Error::OutputRepetition {
            partial: Some(partial),
            ..
        }) => Ok(*partial),
        result => result,
    }
}

fn ran_within(last_run: Option<Instant>, now: Instant, window: Duration) -> bool {
    last_run.is_some_and(|last_run| now.saturating_duration_since(last_run) < window)
}

/// Removes a subtitle credit the model appended to the end of the clip
/// (`subtitle_credits`, #1141) from the text, the last segments, and their
/// words.
fn drop_subtitle_credit(transcript: &mut transcribe_cpp::Transcript) {
    use crate::subtitle_credits::{strip, token_count};
    let Some((text, removed_from_text)) = strip(&transcript.text) else {
        return;
    };
    transcript.text = text;
    let mut removed = if transcript.segments.is_empty() {
        removed_from_text
    } else {
        0
    };
    while let Some(segment) = transcript.segments.last_mut() {
        let Some((kept, tokens)) = strip(&segment.text) else {
            break;
        };
        if kept.is_empty() {
            let first = (segment.first_word.max(0) as usize).min(transcript.words.len());
            transcript.words.truncate(first);
            transcript.segments.pop();
            continue;
        }
        segment.text = kept;
        removed = tokens;
        break;
    }
    if transcript.segments.is_empty() && transcript.text.is_empty() {
        transcript.words.clear();
        return;
    }
    // Words of a segment that kept some speech, or of an unsegmented run.
    let last_segment = transcript.segments.last_mut();
    let floor = last_segment
        .as_ref()
        .map_or(0, |segment| segment.first_word.max(0) as usize);
    let mut popped = 0;
    while removed > 0 && transcript.words.len() > floor {
        let Some(word) = transcript.words.last() else {
            break;
        };
        let tokens = token_count(&word.text);
        if tokens > removed {
            break;
        }
        removed -= tokens;
        transcript.words.pop();
        popped += 1;
    }
    if let Some(segment) = last_segment {
        segment.n_words = (segment.n_words - popped).max(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunks_times_move_onto_the_clips_timeline_by_sample_position() {
        // Offsets come from where a range starts, never from how many chunks
        // produced text.
        assert_eq!(sample_offset_ms(0), 0);
        assert_eq!(sample_offset_ms(2 * 30 * 16_000), 60_000);
        assert_eq!(sample_offset_ms(1_600), 100);

        let shifted = timed_spans(
            transcribe_cpp::Transcript {
                text: "hello".into(),
                segments: vec![transcribe_cpp::Segment {
                    text: " hello ".into(),
                    t0_ms: 0,
                    t1_ms: 480,
                    ..Default::default()
                }],
                words: vec![
                    transcribe_cpp::Word {
                        text: "hello".into(),
                        t0_ms: 10,
                        t1_ms: 480,
                        ..Default::default()
                    },
                    transcribe_cpp::Word {
                        text: "  ".into(),
                        t0_ms: 500,
                        t1_ms: 600,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            sample_offset_ms(2 * 30 * 16_000),
        );
        assert_eq!(shifted.segments.len(), 1);
        assert_eq!(
            (
                shifted.segments[0].start_ms,
                shifted.segments[0].end_ms,
                shifted.segments[0].text.as_str()
            ),
            (60_000, 60_480, "hello")
        );
        assert_eq!(shifted.words.len(), 1, "an empty word is dropped");
        assert_eq!(
            (shifted.words[0].start_ms, shifted.words[0].end_ms),
            (60_010, 60_480)
        );
    }

    #[test]
    fn long_clip_ranges_cover_every_sample_once_within_the_limit() {
        let samples = vec![0.2_f32; 3 * 16_000 + 123];
        let max_samples = 16_000;
        let ranges = chunk_ranges(&samples, max_samples);
        assert_eq!(ranges.first().map(|range| range.start), Some(0));
        assert_eq!(ranges.last().map(|range| range.end), Some(samples.len()));
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "the ranges are contiguous");
        }
        for range in &ranges {
            assert!(!range.is_empty(), "{range:?}");
            assert!(range.len() <= max_samples, "{range:?}");
        }
        assert_eq!(
            ranges.iter().map(|range| range.len()).sum::<usize>(),
            samples.len()
        );
    }

    #[test]
    fn a_long_clip_is_cut_inside_a_silence_near_the_window_end() {
        // Cohere's 35 s window, and a silent second inside the 5 s a cut is
        // searched for at the end of the first window.
        let max_samples = 35 * 16_000;
        let mut samples = vec![0.25_f32; 44 * 16_000];
        let gap = 31 * 16_000..32 * 16_000;
        samples[gap.clone()].fill(0.0);
        let ranges = chunk_ranges(&samples, max_samples);
        assert_eq!(ranges.len(), 2);
        let cut = ranges[0].end;
        assert!(gap.contains(&cut), "cut at {cut} is outside {gap:?}");
        assert_eq!(ranges[1].start, cut);
    }

    #[test]
    fn a_clip_without_silence_still_cuts_in_the_search_region() {
        let max_samples = 35 * 16_000;
        let samples = vec![0.25_f32; 40 * 16_000];
        let ranges = chunk_ranges(&samples, max_samples);
        assert_eq!(ranges.len(), 2);
        let cut = ranges[0].end;
        assert!(
            (max_samples - QUIET_SEARCH_SAMPLES..max_samples).contains(&cut),
            "cut at {cut} is outside the last 5 s of the first window"
        );
        assert_eq!(
            cut,
            max_samples - QUIET_SPAN_SAMPLES / 2,
            "every span is equally quiet, so the latest one wins"
        );
        assert!(ranges[1].len() <= max_samples);
    }

    #[test]
    fn a_clip_that_fits_one_window_is_not_cut() {
        let samples = vec![0.1_f32; 16_000];
        assert_eq!(chunk_ranges(&samples, 30 * 16_000), vec![0..16_000]);
        assert_eq!(chunk_ranges(&samples, 16_000), vec![0..16_000]);
        assert!(chunk_ranges(&[], 16_000).is_empty());
    }

    #[test]
    fn a_models_timestamp_kind_names_its_finest_level() {
        for (kind, level) in [
            (transcribe_cpp::TimestampKind::Word, TimestampLevel::Word),
            (transcribe_cpp::TimestampKind::Token, TimestampLevel::Word),
            (
                transcribe_cpp::TimestampKind::Segment,
                TimestampLevel::Segment,
            ),
            (transcribe_cpp::TimestampKind::Auto, TimestampLevel::Segment),
            (transcribe_cpp::TimestampKind::None, TimestampLevel::None),
        ] {
            assert_eq!(timestamp_level(kind), level, "{kind:?}");
        }
    }

    #[test]
    fn only_a_completed_run_inside_the_window_counts_as_recent() {
        let now = Instant::now();
        let window = Duration::from_secs(60);
        assert!(!ran_within(None, now, window), "a fresh load has not run");
        assert!(ran_within(Some(now), now, window));
        assert!(ran_within(Some(now - Duration::from_secs(59)), now, window));
        assert!(!ran_within(
            Some(now - Duration::from_secs(60)),
            now,
            window
        ));
    }

    #[test]
    fn a_decode_stopped_for_repeating_keeps_its_text_and_truncation_still_fails() {
        let transcript = |text: &str| {
            Some(Box::new(transcribe_cpp::Transcript {
                text: text.into(),
                ..Default::default()
            }))
        };
        let repeated = keep_repetition_partial(Err(transcribe_cpp::Error::OutputRepetition {
            message: "loop".into(),
            partial: transcript("send it now"),
        }));
        assert_eq!(repeated.unwrap().text, "send it now");
        let truncated = keep_repetition_partial(Err(transcribe_cpp::Error::OutputTruncated {
            message: "budget".into(),
            partial: transcript("send it"),
        }));
        assert!(matches!(
            truncated,
            Err(transcribe_cpp::Error::OutputTruncated { .. })
        ));
    }

    #[test]
    fn a_warmup_counts_when_truncated() {
        assert!(!RunKind::Transcript.counts_when_truncated());
        assert!(RunKind::Warmup.counts_when_truncated());
    }

    #[test]
    fn a_decode_stopped_at_its_output_limit_is_recognised_through_its_context() {
        use eyre::{Context, Result};
        let truncated: Result<()> = Err(transcribe_cpp::Error::OutputTruncated {
            message: "budget".into(),
            partial: None,
        })
        .context("transcription failed");
        assert!(output_truncated(&truncated.unwrap_err()));
        let repeating: Result<()> = Err(transcribe_cpp::Error::OutputRepetition {
            message: "loop".into(),
            partial: None,
        })
        .context("transcription failed");
        assert!(!output_truncated(&repeating.unwrap_err()));
    }

    #[test]
    fn a_whisper_clip_is_trimmed_to_its_speech_plus_a_margin() {
        // 10 s clip, speech from 5 s to 9 s.
        assert_eq!(
            whisper_span(160_000, "whisper", Some(80_000..144_000)),
            72_000..152_000
        );
        // Speech up to an edge keeps that edge.
        assert_eq!(
            whisper_span(160_000, "whisper", Some(0..100_000)),
            0..108_000
        );
        assert_eq!(
            whisper_span(160_000, "whisper", Some(60_000..160_000)),
            52_000..160_000
        );
        // Only Whisper trims, and only with speech bounds from the detector.
        assert_eq!(
            whisper_span(160_000, "cohere_asr", Some(80_000..144_000)),
            0..160_000
        );
        assert_eq!(whisper_span(160_000, "whisper", None), 0..160_000);
    }

    #[test]
    fn a_whisper_clip_is_trimmed_and_padded_to_a_chunks_minimum() {
        // 10 s clip, speech from 9.0 s to 9.2 s. The 500 ms margin leaves
        // 1.2 s, which every chunk pads to the 1.5 s minimum.
        let samples = vec![0.25_f32; 160_000];
        let (audio, start) = trimmed_audio(&samples, "whisper", Some(144_000..147_200));
        assert_eq!(start, 136_000);
        assert_eq!(audio.len(), 24_000);
        assert!(audio[..19_200].iter().all(|&sample| sample == 0.25));
        assert!(audio[19_200..].iter().all(|&sample| sample == 0.0));
        // Another model, no detector, or speech reaching both edges: the clip
        // is borrowed unchanged, with no copy or padding.
        for (architecture, bounds) in [
            ("cohere_asr", Some(144_000..147_200)),
            ("whisper", None),
            ("whisper", Some(0..160_000)),
        ] {
            let label = format!("{architecture} {bounds:?}");
            let (audio, start) = trimmed_audio(&samples, architecture, bounds);
            assert!(matches!(audio, Cow::Borrowed(_)), "{label}");
            assert_eq!((start, audio.len()), (0, samples.len()));
        }
    }

    #[test]
    fn a_trimmed_chunks_times_stay_on_the_clips_timeline() {
        let samples = vec![0.1_f32; 160_000];
        let (audio, start) = trimmed_audio(&samples, "whisper", Some(144_000..147_200));
        assert_eq!(start, 136_000);
        assert_eq!(chunk_offset_ms(start, 0), 8_500);
        let ranges = chunk_ranges(&audio, 16_000);
        assert_eq!(ranges.len(), 2, "the padded clip is cut into two windows");
        for range in &ranges {
            assert_eq!(
                chunk_offset_ms(start, range.start),
                8_500 + sample_offset_ms(range.start),
                "{range:?}"
            );
        }
    }

    #[test]
    fn a_long_clip_skips_only_chunks_the_detector_hears_no_speech_in() {
        use crate::speech_detector::override_for_test;
        let chunk = [0.0; 16_000];
        {
            let _silent = override_for_test(Some(false));
            assert!(!chunk_may_hold_speech(&chunk));
        }
        {
            let _speech = override_for_test(Some(true));
            assert!(chunk_may_hold_speech(&chunk));
        }
        let _unavailable = override_for_test(None);
        assert!(
            chunk_may_hold_speech(&chunk),
            "no detector keeps every chunk"
        );
    }

    #[test]
    fn gpu_sessions_bound_decoder_threads_and_cpu_sessions_keep_library_default() {
        use transcribe_cpp::DeviceType;
        assert_eq!(session_options(DeviceType::Gpu).n_threads, 1);
        assert_eq!(session_options(DeviceType::Igpu).n_threads, 1);
        assert_eq!(session_options(DeviceType::Cpu).n_threads, 0);
        assert_eq!(session_options(DeviceType::Accel).n_threads, 0);
    }

    fn segment(text: &str, first_word: i32, n_words: i32) -> transcribe_cpp::Segment {
        transcribe_cpp::Segment {
            text: text.into(),
            first_word,
            n_words,
            ..Default::default()
        }
    }

    fn transcript(segments: &[&str]) -> transcribe_cpp::Transcript {
        let mut transcript = transcribe_cpp::Transcript::default();
        let mut first_word = 0;
        for text in segments {
            let n_words = text.split_whitespace().count() as i32;
            transcript.segments.push(segment(text, first_word, n_words));
            for word in text.split_whitespace() {
                transcript.words.push(transcribe_cpp::Word {
                    text: word.into(),
                    ..Default::default()
                });
            }
            first_word += n_words;
        }
        transcript.text = segments.join(" ");
        transcript
    }

    fn words(transcript: &transcribe_cpp::Transcript) -> Vec<&str> {
        transcript
            .words
            .iter()
            .map(|word| word.text.as_str())
            .collect()
    }

    #[test]
    fn a_trailing_subtitle_credit_is_dropped_from_text_segments_and_words() {
        let mut own = transcript(&["On se voit demain.", "Sous-titrage Société Radio-Canada"]);
        drop_subtitle_credit(&mut own);
        assert_eq!(own.text, "On se voit demain.");
        assert_eq!(own.segments.len(), 1);
        assert_eq!(words(&own), ["On", "se", "voit", "demain."]);

        let mut joined = transcript(&["On se voit demain. Sous-titrage Société Radio-Canada"]);
        drop_subtitle_credit(&mut joined);
        assert_eq!(joined.text, "On se voit demain.");
        assert_eq!(joined.segments[0].text, "On se voit demain.");
        assert_eq!(joined.segments[0].n_words, 4);
        assert_eq!(words(&joined), ["On", "se", "voit", "demain."]);
    }

    #[test]
    fn a_model_without_detection_needs_an_explicit_language() {
        let capabilities = |detects: bool| Capabilities {
            native_sample_rate: 16_000,
            languages: Vec::new(),
            translate_target_languages: Vec::new(),
            max_timestamp_kind: transcribe_cpp::TimestampKind::None,
            supports_language_detect: detects,
            supports_translate: false,
            supports_streaming: false,
            supports_spec_decode: false,
            max_audio_ms: 0,
        };
        let path = std::path::Path::new("model.gguf");
        assert!(
            resolve_language(path, None, &capabilities(false))
                .unwrap_err()
                .to_string()
                .contains("does not advertise")
        );
        assert_eq!(
            resolve_language(path, Some("en"), &capabilities(false)).unwrap(),
            Some("en".to_string())
        );
        assert_eq!(
            resolve_language(path, None, &capabilities(true)).unwrap(),
            None
        );
        // "auto" is treated as no hint, matching the catalog's AUTO_LANGUAGE.
        assert_eq!(
            resolve_language(path, Some("auto"), &capabilities(true)).unwrap(),
            None
        );
    }
}
