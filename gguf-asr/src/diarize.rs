//! Standalone speaker diarization (who spoke when) on a second GGUF model,
//! merged into ASR transcripts after the fact.
//!
//! transcribe-cpp 0.4.1 runs independent diarize models (`diarize_session()`);
//! the directory's one is NVIDIA's streaming Sortformer 4-spk (≤4 speakers,
//! English-trained). A diarize run's `speaker_id` is only self-consistent
//! **within one run**: overlapping runs may number the same person
//! differently, so callers diarize one window and keep its turns on that
//! window's timeline (the unified-layer backend re-runs a rolling session
//! window at every commit to keep ids stable across segments).
//!
//! The library's merge contract is timestamps: turns carry millisecond
//! spans, the ASR side carries timed rows, and [`assign_speakers`] joins
//! them. A model without any timing (qwen3-asr answers exactly one segment
//! per clip however many people spoke in it) has no rows to join, so its
//! clips go through [`split_transcript`] instead: the clip is cut where the
//! turns say the speaker changed and each piece is transcribed on its own,
//! one speaker-attributed segment per piece. A timed model keeps its own
//! rows and only re-cuts the segments that straddle a change
//! ([`resplit_mixed_segments`]); [`SpeakerSplit`] picks the strategy, with
//! [`speaker_split_for`] as the header-only guess before a load.

use std::path::Path;

use eyre::{Context, Result};
use transcribe_cpp::{
    Backend, DiarizeExtension, DiarizeOptions, Model, ModelOptions, SortformerDiarizeOptions,
    SortformerPreset,
};

use crate::gguf_meta::GgufMeta;
use crate::gguf_session::{QUIET_SPAN_SAMPLES, quietest_span, sample_offset_ms};
use crate::local_transcriber::{Device, init_backends};
use crate::transcript::{Transcript, TranscriptSegment};

/// One speaker turn: 1-based `speaker` over `[start_ms, end_ms]`, as one
/// diarize run reported it — relative to **that run's pcm**, not the session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerTurn {
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: u32,
    /// The turn's probability mass where a model reports one. Standalone
    /// Sortformer runs leave it unset (NaN): the native product thresholds
    /// per-frame probabilities internally and emits binary turns, so this is
    /// only a deterministic tie-break for [`assign_speakers`].
    pub p: f32,
}

/// The engine's diarize model: a second native model next to the ASR one,
/// with its own compute lock, loaded and run only on the engine thread.
pub struct Diarizer {
    session: transcribe_cpp::DiarizeSession,
    /// `speaker` is in `[1, max_speakers]`; the truth after load.
    max_speakers: i32,
}

impl Diarizer {
    /// Loads a diarize model — the library's own flow, no extra gating: load
    /// the weights, ask `diarize_info` (the truth: a model without the
    /// DIARIZE role answers an error here), open one session for its
    /// lifetime. A GPU load failure retries on the CPU, as the ASR model's
    /// loader does.
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let backend = match init_backends(device)? {
            Backend::Cpu => Backend::Cpu,
            _ => match Self::load_on(path, Backend::Auto) {
                Ok(diarizer) => return Ok(diarizer),
                Err(error) => {
                    tracing::warn!(
                        error = format!("{error:#}"),
                        "could not load the diarize model on the GPU; loading it on the CPU"
                    );
                    Backend::Cpu
                }
            },
        };
        Self::load_on(path, backend)
    }

    fn load_on(path: &Path, backend: Backend) -> Result<Self> {
        let model = Model::load_with(
            path,
            &ModelOptions {
                backend,
                ..Default::default()
            },
        )
        .with_context(|| format!("could not load the diarize model {}", path.display()))?;
        let info = model
            .diarize_info()
            .with_context(|| format!("{} has no diarize role", path.display()))?;
        let session = model
            .diarize_session()
            .context("could not open a diarize session")?;
        tracing::info!(
            backend = ?backend,
            max_speakers = info.max_speakers,
            "loaded diarize model"
        );
        Ok(Self {
            session,
            max_speakers: info.max_speakers,
        })
    }

    /// The largest speaker id a run may report.
    pub fn max_speakers(&self) -> i32 {
        self.max_speakers
    }

    /// Diarizes one 16 kHz mono clip; turns are relative to `pcm`'s start.
    /// Runs on the engine thread, so it never overlaps an ASR run on the GPU.
    /// The operating point is the library's offline-file preset
    /// (`VeryHighLatency`, ~30.4 s lookahead) — our runs are batch windows,
    /// not realtime feeds; input length is unbounded (the native forward
    /// chunks internally).
    pub fn run(&mut self, pcm: &[f32]) -> Result<Vec<SpeakerTurn>> {
        let options = DiarizeOptions {
            family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
                preset: Some(SortformerPreset::VeryHighLatency),
            })),
        };
        let segments = self
            .session
            .run(pcm, &options)
            .context("diarize run failed")?;
        Ok(segments
            .into_iter()
            .filter_map(|segment| {
                let speaker = u32::try_from(segment.speaker_id).ok()?;
                (segment.speaker_id > 0).then_some(SpeakerTurn {
                    start_ms: segment.t0_ms,
                    end_ms: segment.t1_ms,
                    speaker,
                    p: segment.p,
                })
            })
            .collect())
    }
}

/// Merges diarize `turns` into a transcript's segments: each segment takes
/// the speaker whose turns overlap it longest in total, ties going to the
/// larger probability mass. Both timelines must share one origin (the clip
/// the turns were run on and the clip the segments were transcribed from are
/// the same, or the caller shifted them to agree). Words never get a speaker;
/// a whole-span speaker is the reader's roll-up, not this merge's.
pub fn assign_speakers(transcript: &mut Transcript, turns: &[SpeakerTurn]) {
    for segment in &mut transcript.segments {
        // No overlapping turn keeps the model's own label, if it had one.
        segment.speaker =
            speaker_for(segment.start_ms, segment.end_ms, turns).or(segment.speaker);
    }
}

/// The speaker `turns` attribute to `[start_ms, end_ms]`: largest total
/// overlap wins, `p` breaks exact ties; no overlap, `None`.
fn speaker_for(start_ms: i64, end_ms: i64, turns: &[SpeakerTurn]) -> Option<u32> {
    // (speaker, total overlap ms, total p); at most `max_speakers` rows.
    let mut totals: Vec<(u32, i64, f32)> = Vec::new();
    for turn in turns {
        let overlap = turn.end_ms.min(end_ms) - turn.start_ms.max(start_ms);
        if overlap <= 0 {
            continue;
        }
        // NaN is "unset": it must never outrank a real probability.
        let p = if turn.p.is_nan() { 0.0 } else { turn.p };
        match totals
            .iter_mut()
            .find(|(speaker, _, _)| *speaker == turn.speaker)
        {
            Some((_, overlap_sum, p_sum)) => {
                *overlap_sum += overlap;
                *p_sum += p;
            }
            None => totals.push((turn.speaker, overlap, p)),
        }
    }
    totals
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)))
        .map(|(speaker, _, _)| speaker)
}

/// How much audio a speaker piece spans at minimum: below this a turn change
/// is more plausibly a diarize flicker than a real interjection, so the piece
/// merges into its neighbour instead of standing alone.
const MIN_PIECE_MS: i64 = 600;
/// How far a cut between two pieces may slide to land in the quietest span,
/// so it does not slice a word in half: 250 ms each way.
const SNAP_RADIUS_SAMPLES: usize = 250 * 16;

/// One speaker-homogeneous piece of a clip: the dominant diarize speaker over
/// `[start_ms, end_ms]`, `None` where no turn reached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerChunk {
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: Option<u32>,
}

/// Cuts `[0, len_ms]` into speaker-homogeneous pieces by `turns`: every
/// stretch between turn boundaries takes its dominant speaker
/// ([`speaker_for`]), stretches of one speaker merge, and pieces shorter than
/// `min_piece_ms` — turn flicker, brief unattributed gaps — merge into their
/// previous neighbour (a short head into the next one). Degenerate turns
/// never contribute; no turn at all leaves one unattributed piece.
pub fn speaker_chunks(len_ms: i64, turns: &[SpeakerTurn], min_piece_ms: i64) -> Vec<SpeakerChunk> {
    if len_ms <= 0 {
        return Vec::new();
    }
    let mut bounds: Vec<i64> = Vec::with_capacity(2 + 2 * turns.len());
    bounds.push(0);
    bounds.push(len_ms);
    for turn in turns {
        if turn.end_ms > turn.start_ms {
            for point in [turn.start_ms, turn.end_ms] {
                if point > 0 && point < len_ms {
                    bounds.push(point);
                }
            }
        }
    }
    bounds.sort_unstable();
    bounds.dedup();

    let mut pieces: Vec<SpeakerChunk> = Vec::new();
    for span in bounds.windows(2) {
        let speaker = speaker_for(span[0], span[1], turns);
        match pieces.last_mut() {
            Some(piece) if piece.speaker == speaker => piece.end_ms = span[1],
            _ => pieces.push(SpeakerChunk {
                start_ms: span[0],
                end_ms: span[1],
                speaker,
            }),
        }
    }

    // A piece too short to stand on its own joins a neighbour, keeping the
    // neighbour's speaker: a blip is the neighbour's flicker more often than
    // its own turn. Merging only ever extends spans, so this settles.
    while pieces.len() > 1 {
        let Some(index) = (0..pieces.len())
            .find(|&index| pieces[index].end_ms - pieces[index].start_ms < min_piece_ms)
        else {
            break;
        };
        if index > 0 {
            let end = pieces[index].end_ms;
            pieces.remove(index);
            pieces[index - 1].end_ms = end;
        } else {
            let start = pieces[0].start_ms;
            pieces.remove(0);
            pieces[0].start_ms = start;
        }
    }
    // Merging can make same-speaker pieces adjacent; `dedup_by` passes the
    // pair with `a` as the later element — the one it drops — so the earlier
    // `b` survives by taking the later's end.
    pieces.dedup_by(|a, b| {
        if a.speaker == b.speaker {
            b.end_ms = a.end_ms;
            true
        } else {
            false
        }
    });
    pieces
}

/// Where a cut at `at` slides to: the midpoint of the quietest 100 ms within
/// `radius` samples of it — the same search a long clip's chunk cuts use —
/// so a speaker cut lands in silence instead of mid-word.
fn snap_to_quiet(pcm: &[f32], at: usize, radius: usize) -> usize {
    if pcm.is_empty() {
        return 0;
    }
    let start = at.saturating_sub(radius).min(pcm.len() - 1);
    let end = (at + radius).min(pcm.len());
    if end <= start {
        return at.min(pcm.len());
    }
    start + quietest_span(&pcm[start..end], QUIET_SPAN_SAMPLES)
}

/// The sample ranges `pieces` cut `pcm` into: every internal boundary slides
/// into the quietest nearby span ([`snap_to_quiet`]) while the outer edges
/// stay at the clip's, so the ranges tile the clip without gaps or overlap
/// and each carries its piece. Degenerate inputs yield empty ranges.
fn piece_ranges(
    pcm: &[f32],
    pieces: &[SpeakerChunk],
) -> Vec<(std::ops::Range<usize>, SpeakerChunk)> {
    let mut ranges = Vec::with_capacity(pieces.len());
    let mut start = 0;
    for (index, piece) in pieces.iter().enumerate() {
        // The end of one piece is the start of the next: one snapped cut,
        // shared, so the pieces tile the clip without gaps or overlap.
        let end = if index + 1 < pieces.len() {
            let boundary = (piece.end_ms.max(0) as usize * 16).min(pcm.len());
            snap_to_quiet(pcm, boundary, SNAP_RADIUS_SAMPLES)
                .max(start + 1)
                .min(pcm.len())
        } else {
            pcm.len()
        };
        ranges.push((start..end, *piece));
        start = end;
    }
    ranges
}

/// Transcribes a clip piecewise, one piece per [`SpeakerChunk`]: the flow for
/// models without any timing, whose transcripts hold no rows to merge turns
/// into (see the module doc). The cut between two pieces slides into the
/// quietest nearby span, every piece is transcribed by `transcribe`, and each
/// non-empty piece becomes one speaker-attributed segment spanning the audio
/// actually transcribed; empty pieces are skipped. The text joins exactly as
/// an untimed run joins it, and `language` is the first piece's.
pub fn split_transcript(
    pcm: &[f32],
    turns: &[SpeakerTurn],
    mut transcribe: impl FnMut(&[f32]) -> Result<Transcript>,
) -> Result<Transcript> {
    let pieces = speaker_chunks(sample_offset_ms(pcm.len()), turns, MIN_PIECE_MS);
    let mut transcript = Transcript::default();
    for (range, piece) in piece_ranges(pcm, &pieces) {
        if range.is_empty() {
            continue;
        }
        let transcribed = transcribe(&pcm[range.start..range.end])?;
        let text = transcribed.text.trim();
        if !text.is_empty() {
            if transcript.language.is_none() {
                transcript.language = transcribed.language;
            }
            transcript.segments.push(TranscriptSegment {
                start_ms: sample_offset_ms(range.start),
                end_ms: sample_offset_ms(range.end),
                text: text.to_owned(),
                speaker: piece.speaker,
            });
        }
    }
    transcript.text = transcript
        .segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(transcript)
}

/// Re-splits only the transcript segments that straddle a diarize speaker
/// change: each straddling segment's span is partitioned by the turns
/// ([`speaker_chunks`]), the cuts slide into silence as
/// [`split_transcript`]'s do, and the pieces re-transcribe — a piece's own
/// rows (times moved onto the clip timeline, speaker stamped; a piece is
/// speaker-homogeneous by construction) replace the segment, with a span
/// row synthesized when the piece reports none. Segments holding one
/// speaker stay as the model produced them; a failed piece keeps its
/// segment whole; words inside a re-split span are replaced by the pieces'
/// own. Returns how many segments were re-split.
pub fn resplit_mixed_segments(
    pcm: &[f32],
    transcript: &mut Transcript,
    turns: &[SpeakerTurn],
    mut transcribe: impl FnMut(&[f32]) -> Result<Transcript>,
) -> usize {
    let mut resplit = 0;
    let mut segments = Vec::with_capacity(transcript.segments.len());
    let mut piece_words = Vec::new();
    let mut spans: Vec<(i64, i64)> = Vec::new();
    for (index, segment) in std::mem::take(&mut transcript.segments)
        .into_iter()
        .enumerate()
    {
        // Turns onto the segment's own timeline: a change at a segment edge
        // never lands inside it, so only true straddlers partition.
        let local: Vec<SpeakerTurn> = turns
            .iter()
            .map(|turn| SpeakerTurn {
                start_ms: turn.start_ms - segment.start_ms,
                end_ms: turn.end_ms - segment.start_ms,
                speaker: turn.speaker,
                p: turn.p,
            })
            .collect();
        let pieces = speaker_chunks(segment.end_ms - segment.start_ms, &local, MIN_PIECE_MS);
        let mut speakers: Vec<u32> = Vec::new();
        for piece in &pieces {
            if let Some(speaker) = piece.speaker
                && !speakers.contains(&speaker)
            {
                speakers.push(speaker);
            }
        }
        if speakers.len() < 2 {
            segments.push(segment);
            continue;
        }
        let mut rows: Vec<TranscriptSegment> = Vec::new();
        let mut failed = false;
        // Pieces push words as they go; a segment kept whole must take its
        // pieces' words back out, or they duplicate the model's own.
        let words_before = piece_words.len();
        let start = (segment.start_ms.max(0) as usize * 16).min(pcm.len());
        let end = (segment.end_ms.max(0) as usize * 16).min(pcm.len());
        if end > start {
            let span = &pcm[start..end];
            for (range, piece) in piece_ranges(span, &pieces) {
                if range.is_empty() {
                    continue;
                }
                match transcribe(&span[range.clone()]) {
                    Ok(transcribed) => {
                        let offset_ms = sample_offset_ms(start + range.start);
                        let text = transcribed.text.trim();
                        if !text.is_empty() && transcribed.segments.is_empty() {
                            rows.push(TranscriptSegment {
                                start_ms: offset_ms,
                                end_ms: sample_offset_ms(start + range.end),
                                text: text.to_owned(),
                                speaker: piece.speaker,
                            });
                        }
                        // A piece that reports its own rows keeps them, on
                        // the clip's timeline and under the piece's speaker.
                        for mut row in transcribed.segments {
                            if row.text.trim().is_empty() {
                                continue;
                            }
                            row.end_ms = row.end_ms.max(row.start_ms) + offset_ms;
                            row.start_ms += offset_ms;
                            row.speaker = piece.speaker;
                            rows.push(row);
                        }
                        for mut word in transcribed.words {
                            word.start_ms += offset_ms;
                            word.end_ms += offset_ms;
                            piece_words.push(word);
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            segment = index,
                            error = format!("{error:#}"),
                            "a piece of a mixed segment failed; keeping the model's segment"
                        );
                        failed = true;
                        break;
                    }
                }
            }
        }
        if failed || rows.is_empty() {
            piece_words.truncate(words_before);
            segments.push(segment);
            continue;
        }
        resplit += 1;
        spans.push((segment.start_ms, segment.end_ms));
        segments.extend(rows);
    }
    transcript.segments = segments;
    if resplit > 0 {
        // Words a re-split span owned are replaced by the pieces' own; words
        // outside every span keep their times. Time order keeps the merge
        // honest, and the text follows the rows it now has.
        for (start_ms, end_ms) in &spans {
            transcript
                .words
                // A word belongs to the span its midpoint falls in: model
                // word times overrun segment bounds by a few ms, and a
                // containment test would keep a straddler the pieces
                // transcribe again.
                .retain(|word| {
                    let mid = word.start_ms + (word.end_ms - word.start_ms) / 2;
                    !(mid >= *start_ms && mid < *end_ms)
                });
        }
        transcript.words.extend(piece_words);
        transcript.words.sort_by_key(|word| word.start_ms);
        transcript.text = transcript
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    }
    resplit
}

/// How diarize turns reshape a transcript, beyond stamping each of the
/// model's own segments: the speaker-split enhancement. `Auto` (the
/// default) resolves against the loaded engine's model
/// ([`crate::Engine::speaker_split`], the runtime truth); the header-only
/// guess for before the load is [`speaker_split_for`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpeakerSplit {
    #[default]
    Auto,
    /// Turns only stamp per-segment speakers — the merge without the
    /// enhancement.
    Off,
    /// The whole clip transcribes piecewise, one piece per speaker chunk
    /// ([`split_transcript`]): for models without any timing, whose
    /// transcripts hold one segment per clip however many people spoke.
    Pieces,
    /// The model's segments stay; only those straddling a speaker change
    /// re-transcribe piecewise ([`resplit_mixed_segments`]). Safe on models
    /// without timing too: their single whole-clip segment is a straddler
    /// whenever two speakers appear, so this degrades to [`Self::Pieces`].
    MixedSegments,
}

/// The speaker-split strategy a GGUF's header claims its family needs — the
/// pre-load guess beside [`crate::Engine::speaker_split`]'s runtime truth
/// (the same claim/truth split as [`crate::gguf_meta`]'s streaming flag and
/// `Engine::streams`). Families whose transcribe-cpp capabilities report no
/// timestamp kind answer [`SpeakerSplit::Pieces`]; everything timed or
/// unknown answers [`SpeakerSplit::MixedSegments`], which degrades
/// gracefully on the timing-less.
pub fn speaker_split_for(meta: &GgufMeta) -> SpeakerSplit {
    match meta.architecture.as_str() {
        // transcribe-cpp 0.4.1 family invariants: max_timestamp_kind = NONE
        // (each arch's capabilities.cpp; names as the arch registry has
        // them — the module dirs "cohere"/"granite_nar" register as
        // "cohere_asr"/"granite_speech_nar").
        "canary"
        | "canary_qwen"
        | "cohere_asr"
        | "funasr_nano"
        | "granite_speech5_ctc"
        | "granite_speech_nar"
        | "moonshine"
        | "moonshine_streaming"
        | "qwen3_asr"
        | "sensevoice"
        | "voxtral"
        | "voxtral_realtime" => SpeakerSplit::Pieces,
        // whisper (segment timings), parakeet (token), and families this
        // build has not met: err toward keeping the model's own rows and
        // still split a straddler.
        _ => SpeakerSplit::MixedSegments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::TranscriptSegment;

    fn turn(start_ms: i64, end_ms: i64, speaker: u32) -> SpeakerTurn {
        SpeakerTurn {
            start_ms,
            end_ms,
            speaker,
            p: 1.0,
        }
    }

    fn segments(spans: &[(i64, i64)]) -> Vec<TranscriptSegment> {
        spans
            .iter()
            .map(|&(start_ms, end_ms)| TranscriptSegment {
                start_ms,
                end_ms,
                text: String::new(),
                speaker: None,
            })
            .collect()
    }

    /// 每段取重叠时长最大的 speaker;平手按 p 加权;无重叠为 None。
    #[test]
    fn the_longest_overlap_wins_and_ties_go_to_probability() {
        // 段 0-1000:说话人 1 重叠 900,说话人 2 重叠 500 → 1。
        // 段 1500-2500:只有说话人 2 → 2。
        let turns = [turn(0, 900, 1), turn(700, 2500, 2)];
        let mut transcript = Transcript {
            segments: segments(&[(0, 1_000), (1_500, 2_500)]),
            ..Transcript::default()
        };
        assign_speakers(&mut transcript, &turns);
        assert_eq!(transcript.segments[0].speaker, Some(1));
        assert_eq!(transcript.segments[1].speaker, Some(2));
    }

    #[test]
    fn an_exact_overlap_tie_goes_to_the_larger_p() {
        let mut transcript = Transcript {
            segments: segments(&[(0, 1_000)]),
            ..Transcript::default()
        };
        let mut loud = turn(0, 600, 1);
        loud.p = 0.9;
        let mut quiet = turn(400, 1_000, 2);
        quiet.p = 0.2;
        assign_speakers(&mut transcript, &[loud, quiet]);
        assert_eq!(transcript.segments[0].speaker, Some(1));
        assign_speakers(&mut transcript, &[quiet, loud]);
        assert_eq!(transcript.segments[0].speaker, Some(1), "与顺序无关");
    }

    #[test]
    fn a_segment_no_turn_touches_has_no_speaker() {
        let mut transcript = Transcript {
            segments: segments(&[(0, 1_000), (2_000, 3_000)]),
            ..Transcript::default()
        };
        assign_speakers(&mut transcript, &[turn(2_100, 2_900, 1)]);
        assert_eq!(transcript.segments[0].speaker, None);
        assert_eq!(transcript.segments[1].speaker, Some(1));
    }

    #[test]
    fn one_turn_per_speaker_overlaps_sum_before_the_winner_is_picked() {
        // 说话人 1 两段各重叠 300(合计 600),说话人 2 一段重叠 500 → 1。
        let turns = [turn(0, 300, 1), turn(700, 1_000, 1), turn(300, 800, 2)];
        let mut transcript = Transcript {
            segments: segments(&[(0, 1_000)]),
            ..Transcript::default()
        };
        assign_speakers(&mut transcript, &turns);
        assert_eq!(transcript.segments[0].speaker, Some(1));
    }

    #[test]
    fn a_single_speaker_labels_every_segment_it_overlaps() {
        let turns = [turn(0, 4_000, 3)];
        let mut transcript = Transcript {
            segments: segments(&[(0, 1_000), (1_000, 2_000), (3_500, 4_500)]),
            ..Transcript::default()
        };
        assign_speakers(&mut transcript, &turns);
        assert!(
            transcript.segments[..2]
                .iter()
                .all(|segment| segment.speaker == Some(3))
        );
        assert_eq!(transcript.segments[2].speaker, Some(3), "重叠 500ms 也归属");
    }

    #[test]
    fn degenerate_turns_and_spans_never_panic_or_win() {
        // 起点在终点之后的 turn、零长段:都不贡献归属。
        let turns = [turn(2_000, 1_000, 1), turn(0, 1_000, 2)];
        let mut transcript = Transcript {
            segments: segments(&[(1_500, 1_500), (0, 1_000)]),
            ..Transcript::default()
        };
        assign_speakers(&mut transcript, &turns);
        assert_eq!(transcript.segments[0].speaker, None);
        assert_eq!(transcript.segments[1].speaker, Some(2));
    }

    #[test]
    fn empty_turns_leave_every_segment_unattributed() {
        let mut transcript = Transcript {
            segments: segments(&[(0, 1_000)]),
            ..Transcript::default()
        };
        assign_speakers(&mut transcript, &[]);
        assert_eq!(transcript.segments[0].speaker, None);
    }

    fn chunk(start_ms: i64, end_ms: i64, speaker: Option<u32>) -> SpeakerChunk {
        SpeakerChunk {
            start_ms,
            end_ms,
            speaker,
        }
    }

    /// 交替的两个说话人切成两片;重叠交接段两边全覆盖 → `speaker_for`
    /// 平手(`max_by` 取后注册者)归第二个说话人。
    #[test]
    fn alternating_speakers_cut_into_two_pieces() {
        let turns = [turn(0, 2_000, 1), turn(2_000, 4_000, 2)];
        assert_eq!(
            speaker_chunks(4_000, &turns, 600),
            vec![chunk(0, 2_000, Some(1)), chunk(2_000, 4_000, Some(2))]
        );
        let overlapping = [turn(0, 2_400, 1), turn(1_400, 4_000, 2)];
        assert_eq!(
            speaker_chunks(4_000, &overlapping, 600),
            vec![chunk(0, 1_400, Some(1)), chunk(1_400, 4_000, Some(2))]
        );
    }

    /// 短于 min 的翻转/空隙并入前片,同说话人相邻段合并且整段归一个人。
    #[test]
    fn short_flickers_and_gaps_merge_into_the_previous_piece() {
        let turns = [
            turn(0, 2_000, 1),
            turn(2_100, 2_500, 2), // 400ms 翻转
            turn(2_600, 4_000, 1),
        ];
        let pieces = speaker_chunks(4_000, &turns, 600);
        assert_eq!(pieces, vec![chunk(0, 4_000, Some(1))]);
    }

    /// 短头部(无前片)并入后一片,保留后片的说话人。
    #[test]
    fn a_short_head_merges_into_the_next_piece() {
        let turns = [turn(0, 300, 1), turn(300, 4_000, 2)];
        let pieces = speaker_chunks(4_000, &turns, 600);
        assert_eq!(pieces, vec![chunk(0, 4_000, Some(2))]);
    }

    /// 长(≥min)无归属空隙自成一 None 片,不被吞并。
    #[test]
    fn a_long_unattributed_gap_stands_on_its_own() {
        let turns = [turn(0, 1_000, 1), turn(2_000, 3_000, 2)];
        let pieces = speaker_chunks(3_000, &turns, 600);
        assert_eq!(
            pieces,
            vec![
                chunk(0, 1_000, Some(1)),
                chunk(1_000, 2_000, None),
                chunk(2_000, 3_000, Some(2))
            ]
        );
    }

    /// 空 turns / 退化 turns / 非正长度:安全返回。
    #[test]
    fn degenerate_inputs_return_whole_or_nothing() {
        assert_eq!(speaker_chunks(0, &[turn(0, 100, 1)], 600), Vec::new());
        assert_eq!(speaker_chunks(1_000, &[], 600), vec![chunk(0, 1_000, None)]);
        assert_eq!(
            speaker_chunks(1_000, &[turn(2_000, 1_000, 1)], 600),
            vec![chunk(0, 1_000, None)]
        );
    }

    /// 越界 turns 被 clamp,不产生 0 长或越界片(段内部分 ≥ min 才立得住)。
    #[test]
    fn turns_outside_the_clip_are_clamped_away() {
        let turns = [turn(-500, 900, 1), turn(1_200, 5_000, 2)];
        assert_eq!(
            speaker_chunks(2_000, &turns, 600),
            vec![chunk(0, 1_200, Some(1)), chunk(1_200, 2_000, Some(2))]
        );
    }

    /// 切点滑进 ±radius 内最静的 100ms 窗中心。
    #[test]
    fn a_cut_slides_into_the_quietest_nearby_span() {
        // 3s 音频:整体有声,1.0s–1.1s(16000–17600 样本)是静音。
        let mut pcm = vec![0.5; 48_000];
        pcm[16_000..17_600].fill(0.0);
        let cut = snap_to_quiet(&pcm, 16_800, 4_000);
        assert_eq!(cut, 16_800, "静音窗中心即切点");
        // 均匀能量(处处同响)时平手取最靠后的窗:窗口 [28000,36000) 的
        // 最后一个 1600 窗起点 34400,中心 35200。
        let flat = vec![0.3; 48_000];
        assert_eq!(snap_to_quiet(&flat, 32_000, 4_000), 35_200);
        assert_eq!(snap_to_quiet(&[], 100, 4_000), 0);
    }

    /// 分片转写:两片各成一段、自带说话人;文本按片拼接。
    #[test]
    fn split_transcript_attributes_each_piece() {
        let turns = [turn(0, 2_000, 1), turn(2_000, 4_000, 2)];
        let pcm = vec![0.3; 64_000]; // 4s
        let transcript = split_transcript(&pcm, &turns, |piece| {
            Ok(Transcript {
                text: format!("len{}", piece.len()),
                language: Some("en".to_string()),
                ..Transcript::default()
            })
        })
        .expect("分片转写成功");
        assert_eq!(transcript.text, "len35200 len28800");
        assert_eq!(
            transcript
                .segments
                .iter()
                .map(|segment| (segment.start_ms, segment.end_ms, segment.speaker))
                .collect::<Vec<_>>(),
            vec![(0, 2_200, Some(1)), (2_200, 4_000, Some(2))],
            "切点滑动后段界 = 实际音频边界(35200 样本 = 2200ms)"
        );
        assert_eq!(transcript.language, Some("en".to_string()));
    }

    /// 空文本片被跳过,不产生空段;语言取第一个非空片。
    #[test]
    fn empty_pieces_are_skipped() {
        let turns = [turn(0, 2_000, 1), turn(2_000, 4_000, 2)];
        let pcm = vec![0.3; 64_000];
        let transcript = split_transcript(&pcm, &turns, |piece| {
            let empty = piece.len() > 30_000; // 第一片 35200 → 空;第二片 28800 → 有声
            Ok(Transcript {
                text: if empty {
                    String::new()
                } else {
                    "spoken".to_string()
                },
                language: (!empty).then(|| "en".to_string()),
                ..Transcript::default()
            })
        })
        .expect("分片转写成功");
        assert_eq!(transcript.segments.len(), 1);
        assert_eq!(transcript.segments[0].speaker, Some(2));
        assert_eq!(transcript.language, Some("en".to_string()));
    }

    /// 任一片转写失败:整体 Err(引擎侧降级整段归并)。
    #[test]
    fn a_failing_piece_fails_the_whole_split() {
        let turns = [turn(0, 2_000, 1), turn(2_000, 4_000, 2)];
        let pcm = vec![0.3; 64_000];
        let error = split_transcript(&pcm, &turns, |_| Err(eyre::eyre!("piece failed")))
            .expect_err("失败必须传播");
        assert!(format!("{error:#}").contains("piece failed"));
    }

    /// 跨换人点的段被替换为逐片行(平移回段轴、钉片 speaker);干净段、
    /// 区域外 words 原样保留。
    #[test]
    fn resplit_replaces_only_straddling_segments() {
        let turns = [turn(0, 2_600, 1), turn(2_600, 4_000, 2)];
        let pcm = vec![0.3; 64_000]; // 4s
        let mut transcript = Transcript {
            text: "clean mixed".to_string(),
            language: Some("en".to_string()),
            segments: segments(&[(0, 2_000), (2_000, 4_000)]),
            words: segments(&[(100, 200), (2_500, 2_600)]),
        };
        transcript.segments[0].text = "clean".into();
        transcript.segments[1].text = "mixed".into();
        transcript.words[0].text = "a".into();
        transcript.words[1].text = "b".into();
        // 引擎流程:归并先给每段盖 speaker 章,重切在其后
        assign_speakers(&mut transcript, &turns);
        let resplit = resplit_mixed_segments(&pcm, &mut transcript, &turns, |piece| {
            Ok(Transcript {
                text: format!("t{}", piece.len()),
                ..Transcript::default()
            })
        });
        assert_eq!(resplit, 1, "只有第二段跨了 2600ms 的换人点");
        assert_eq!(
            transcript
                .segments
                .iter()
                .map(|segment| {
                    (
                        segment.start_ms,
                        segment.end_ms,
                        segment.text.as_str(),
                        segment.speaker,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                (0, 2_000, "clean", Some(1)),
                // 段内片界 600ms 吸附到 800ms(平手取最靠后的静窗)
                (2_000, 2_800, "t12800", Some(1)),
                (2_800, 4_000, "t19200", Some(2)),
            ]
        );
        // 重切区域内的 word 被丢弃(片转写没给),区域外保留;text 重建
        assert_eq!(
            transcript
                .words
                .iter()
                .map(|word| (word.start_ms, word.end_ms, word.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(100, 200, "a")]
        );
        assert_eq!(transcript.text, "clean t12800 t19200");
        assert_eq!(transcript.language, Some("en".to_string()), "语言不被触碰");
    }

    /// 换人点恰好落在段边界:两段都不混合,转写闭包一次都不该被调。
    #[test]
    fn a_change_at_a_segment_edge_does_not_resplit() {
        let turns = [turn(0, 2_000, 1), turn(2_000, 4_000, 2)];
        let pcm = vec![0.3; 64_000];
        let mut transcript = Transcript {
            text: "a b".to_string(),
            language: None,
            segments: segments(&[(0, 2_000), (2_000, 4_000)]),
            words: Vec::new(),
        };
        let before = transcript.clone();
        let resplit = resplit_mixed_segments(&pcm, &mut transcript, &turns, |piece| {
            panic!("no piece should transcribe (got {} samples)", piece.len())
        });
        assert_eq!(resplit, 0);
        assert_eq!(transcript, before, "零重切 = transcript 逐字节原样");
    }

    /// 某片失败只保住那一段(逐段降级),其它混合段照常重切。
    #[test]
    fn a_failing_piece_keeps_its_segment_whole() {
        // 两个混合段:0–4000(2000 换人)与 4000–7000(5500 换人),
        // 片长各异(35200/28800 与 27200/20800),只让第一段首片失败
        let turns = [
            turn(0, 2_000, 1),
            turn(2_000, 4_000, 2),
            turn(4_000, 5_500, 1),
            turn(5_500, 7_000, 2),
        ];
        let pcm = vec![0.3; 112_000]; // 7s
        let mut transcript = Transcript {
            text: "one two".to_string(),
            language: None,
            segments: segments(&[(0, 4_000), (4_000, 7_000)]),
            words: Vec::new(),
        };
        transcript.segments[0].text = "one".into();
        transcript.segments[1].text = "two".into();
        let resplit = resplit_mixed_segments(&pcm, &mut transcript, &turns, |piece| {
            if piece.len() == 35_200 {
                Err(eyre::eyre!("boom"))
            } else {
                Ok(Transcript {
                    text: format!("t{}", piece.len()),
                    ..Transcript::default()
                })
            }
        });
        assert_eq!(resplit, 1, "第一段保原样,第二段重切成功");
        assert_eq!(
            transcript
                .segments
                .iter()
                .map(|segment| (segment.text.as_str(), segment.speaker))
                .collect::<Vec<_>>(),
            vec![("one", None), ("t27200", Some(1)), ("t20800", Some(2))]
        );
    }

    /// 片转写自带 segments/words 时优先采用(平移 + 钉 speaker),合成行
    /// 只在片没有自己的行时兜底。
    #[test]
    fn piece_rows_prefer_the_pieces_own_rows() {
        let turns = [turn(0, 2_000, 1), turn(2_000, 4_000, 2)];
        let pcm = vec![0.3; 64_000];
        let mut transcript = Transcript {
            text: "mixed".to_string(),
            language: None,
            segments: segments(&[(0, 4_000)]),
            words: Vec::new(),
        };
        transcript.segments[0].text = "mixed".into();
        let resplit = resplit_mixed_segments(&pcm, &mut transcript, &turns, |_| {
            Ok(Transcript {
                text: "ignored".to_string(),
                language: None,
                segments: vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 300,
                    text: "own".into(),
                    speaker: None,
                }],
                words: vec![TranscriptSegment {
                    start_ms: 50,
                    end_ms: 150,
                    text: "w".into(),
                    speaker: None,
                }],
            })
        });
        assert_eq!(resplit, 1);
        // 换人点 2000ms 吸附到 2200ms:两片各带自己的行,时间平移、
        // speaker 钉片的主人
        assert_eq!(
            transcript
                .segments
                .iter()
                .map(|segment| (segment.start_ms, segment.end_ms, segment.speaker))
                .collect::<Vec<_>>(),
            vec![(0, 300, Some(1)), (2_200, 2_500, Some(2))]
        );
        assert_eq!(
            transcript
                .words
                .iter()
                .map(|word| (word.start_ms, word.end_ms))
                .collect::<Vec<_>>(),
            vec![(50, 150), (2_250, 2_350)]
        );
    }

    /// meta 快方法:已知无时间戳家族 → Pieces;有时间戳与未知 →
    /// MixedSegments(对无时间戳模型优雅退化,安全默认)。
    #[test]
    fn speaker_split_for_maps_families() {
        let meta = |architecture: &str| GgufMeta {
            architecture: architecture.to_string(),
            streaming: false,
            name: None,
        };
        assert_eq!(speaker_split_for(&meta("qwen3_asr")), SpeakerSplit::Pieces);
        assert_eq!(speaker_split_for(&meta("cohere_asr")), SpeakerSplit::Pieces);
        assert_eq!(
            speaker_split_for(&meta("granite_speech5_ctc")),
            SpeakerSplit::Pieces
        );
        assert_eq!(
            speaker_split_for(&meta("whisper")),
            SpeakerSplit::MixedSegments
        );
        assert_eq!(
            speaker_split_for(&meta("parakeet")),
            SpeakerSplit::MixedSegments
        );
        assert_eq!(
            speaker_split_for(&meta("made_up")),
            SpeakerSplit::MixedSegments,
            "未知家族走安全默认"
        );
    }
}
