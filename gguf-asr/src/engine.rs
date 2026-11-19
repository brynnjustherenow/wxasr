//! The thread that owns the loaded GGUF model and runs all of its inference.
//!
//! A capture that the model can stream is fed while it records, so the
//! finish only decodes the last second or two of audio. Every finish carries
//! the complete clip: a stream that failed, stalled, or produced no text is
//! replaced by an offline transcription of that clip, and a model or device
//! that cannot stream always takes the offline path. Commands are handled in
//! order on one thread, because a live stream holds the model's compute lease.
//!
//! Hex-inspired: Hex's Labs gating (live text, stop phrase, skip pauses,
//! free-memory-when-idle) is not reimplemented here. Partial text is always
//! published for a streaming capture, and long pauses are never cut from a
//! whole clip.

use std::collections::VecDeque;
mod streaming;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use eyre::{Result, eyre};
use transcribe_cpp::CancelToken;

use crate::diarize::{Diarizer, SpeakerSplit, SpeakerTurn};
use crate::local_transcriber::{Device, LocalTranscriber, ModelSpec};
use crate::transcript::Transcript;
use streaming::*;

/// The model the engine owns: the one local GGUF model.
pub(crate) type Model = LocalTranscriber;

/// Transcribes a complete 16 kHz clip offline on the engine thread.
pub type Offline = Box<dyn FnOnce(&mut Model, Vec<f32>) -> Result<Transcript, String> + Send>;

/// Low-priority work on the model. It receives `None` instead of the model
/// when a dictation-class request claimed the engine first.
pub type Background = Box<dyn FnOnce(Option<&mut Model>) + Send>;

/// A healthy finish decodes at most a couple of seconds of buffered audio.
/// Past this the stream is aborted and the clip is transcribed offline.
const STREAM_FINISH_DEADLINE: Duration = Duration::from_millis(1_500);
/// Audio fed past the clip's trimmed end that the stream may still keep.
/// Beyond this, the offline clip is authoritative.
const MAX_OVERFED_SAMPLES: usize = 16_000 / 4;
/// How long a job may run once the engine starts it, before the clip's own
/// share. Time queued behind other work is never counted: the deadline only
/// catches a wedged engine, never a slow one.
const JOB_DEADLINE: Duration = Duration::from_secs(180);
/// Run time allowed per second of audio: two passes on a CPU-only device up
/// to four times slower than real time.
const JOB_SECONDS_PER_AUDIO_SECOND: u32 = 8;
/// How often a waiter whose job has not started checks whether the job ahead
/// of it timed out, leaving the engine wedged.
const QUEUE_POLL: Duration = Duration::from_millis(250);
/// How long dropping the engine waits for its thread. A native run that cannot
/// be aborted is left to finish on its own.
const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(10);

#[derive(Clone, Copy)]
struct Deadlines {
    stream_finish: Duration,
    job: Duration,
    job_seconds_per_audio_second: u32,
    shutdown: Duration,
}

impl Deadlines {
    const PRODUCTION: Self = Self {
        stream_finish: STREAM_FINISH_DEADLINE,
        job: JOB_DEADLINE,
        job_seconds_per_audio_second: JOB_SECONDS_PER_AUDIO_SECOND,
        shutdown: SHUTDOWN_DEADLINE,
    };

    fn job_for(&self, samples: usize) -> Duration {
        self.job
            + Duration::from_secs_f64(samples as f64 / 16_000.0) * self.job_seconds_per_audio_second
    }
}

pub(crate) enum Command {
    Begin {
        stream: u64,
        /// Aborts this stream only.
        cancel: CancelToken,
        /// Where the stream publishes its text after each feed.
        partials: Option<Partials>,
        /// Clip samples of the caller's own sounds at the capture's head: a
        /// pre-roll, a start tone. Speech heard there is never a lost
        /// opening.
        head_ignored: usize,
    },
    Feed {
        stream: u64,
        samples: Vec<f32>,
    },
    Cancel {
        stream: u64,
    },
    Finish {
        stream: Option<u64>,
        clip: Vec<f32>,
        offline: Offline,
        job: Job,
    },
    /// An offline run that waits for an open live stream to end instead of
    /// interrupting it, such as a word-count chunk. Like a finish it is a
    /// dictation-class request, so background work sent before it yields.
    Foreground {
        clip: Vec<f32>,
        offline: Offline,
        job: Job,
    },
    /// A live stream that gives way to dictation: any dictation request
    /// closes it, and while a dictation stream is open it is refused
    /// instead of interrupting it.
    BeginYielding {
        stream: u64,
        cancel: CancelToken,
        yielding: Yielding,
    },
    /// Finalizes a yielding stream, or answers [`PREEMPTED`] at once when it
    /// is no longer open. It never runs anything offline.
    FinishYielding {
        stream: u64,
        reply: SyncSender<Result<String, String>>,
    },
    Background {
        /// The request counter when the work was submitted.
        generation: u64,
        work: Background,
    },
    /// Diarizes a clip on the engine's second model, as its own job with its
    /// own cancellation: a speaker-attributed commit runs this first, then
    /// transcribes the segment with the turns.
    Diarize {
        pcm: Vec<f32>,
        job: Job,
    },
    Shutdown,
}

/// Why a yielding stream ended without its text: dictation needed the engine.
pub const PREEMPTED: &str = "the stream gave way to dictation";

/// A live stream's text so far, as the model reports it after each feed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Partial {
    /// The prefix the model has committed; it only grows within one stream.
    pub stable: String,
    /// The model's volatile hypothesis after `stable`.
    pub tentative: String,
}

/// One stream's latest partial text, shared between the worker that
/// publishes it and the stream's owner. Each stream has its own, so a new
/// capture never reads an older one's text.
#[derive(Clone, Default)]
pub(crate) struct Partials(Arc<Mutex<Published>>);

#[derive(Default)]
struct Published {
    /// Bumped by each change, so an owner can tell what it has not seen.
    seq: u64,
    partial: Option<Partial>,
}

impl Partials {
    /// Publishes `partial`; returns whether it changed the text.
    pub(crate) fn publish(&self, partial: Partial) -> bool {
        let mut published = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if published.partial.as_ref() == Some(&partial) {
            return false;
        }
        published.seq += 1;
        published.partial = Some(partial);
        true
    }

    fn latest(&self) -> Option<Partial> {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .partial
            .clone()
    }

    /// The latest partial when it is newer than `seen`, which it advances.
    fn newer_than(&self, seen: &mut u64) -> Option<Partial> {
        let published = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if published.seq <= *seen {
            return None;
        }
        *seen = published.seq;
        published.partial.clone()
    }
}

/// What the worker shares with a yielding stream's owner: whether the stream
/// is still open on the engine, and its latest partial text.
#[derive(Clone)]
pub(crate) struct Yielding {
    open: Arc<AtomicBool>,
    /// Samples the model has consumed.
    consumed: Arc<AtomicUsize>,
    partials: Partials,
}

impl Yielding {
    fn new() -> Self {
        Self {
            open: Arc::new(AtomicBool::new(true)),
            consumed: Arc::default(),
            partials: Partials::default(),
        }
    }

    fn close(&self) {
        self.open.store(false, Ordering::Release);
    }
}

/// A live stream that gives way to dictation. Unlike a [`LiveStream`] it is
/// not a dictation request: background work sent before it still runs, and
/// any dictation request closes it. Feed it 16 kHz audio, then
/// [`Self::finish`] it or drop it to cancel.
pub struct YieldingStream {
    id: u64,
    cancel: CancelToken,
    yielding: Yielding,
    sent: usize,
    finished: bool,
    commands: Sender<Command>,
}

impl YieldingStream {
    /// Whether the stream is still open on the engine. Once dictation closes
    /// it, its audio must be transcribed some other way.
    pub fn open(&self) -> bool {
        self.yielding.open.load(Ordering::Acquire)
    }

    pub fn feed(&mut self, samples: &[f32]) {
        if samples.is_empty() || !self.open() {
            return;
        }
        self.sent += samples.len();
        let _ = self.commands.send(Command::Feed {
            stream: self.id,
            samples: samples.to_vec(),
        });
    }

    /// Samples sent that the model has not consumed yet. Keeping this small
    /// keeps a dictation request from queueing behind a pile of feeds.
    pub fn in_flight(&self) -> usize {
        self.sent
            .saturating_sub(self.yielding.consumed.load(Ordering::Acquire))
    }

    /// The model's latest text for the audio fed so far.
    pub fn partial(&self) -> Option<Partial> {
        self.yielding.partials.latest()
    }

    /// Finalizes the stream within `deadline`. It fails with [`PREEMPTED`]
    /// when dictation closed it first, and is aborted when the engine does
    /// not reach it in time; either way nothing was lost but this text.
    pub fn finish(mut self, deadline: Duration) -> Result<String, String> {
        self.finished = true;
        if !self.open() {
            return Err(PREEMPTED.into());
        }
        let (reply, receiver) = mpsc::sync_channel(1);
        if self
            .commands
            .send(Command::FinishYielding {
                stream: self.id,
                reply,
            })
            .is_err()
        {
            return Err("transcription engine stopped".into());
        }
        match receiver.recv_timeout(deadline) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                self.cancel.cancel();
                Err("the stream missed its finish deadline".into())
            }
            Err(RecvTimeoutError::Disconnected) => Err("transcription engine stopped".into()),
        }
    }
}

impl Drop for YieldingStream {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.commands.send(Command::Cancel { stream: self.id });
        }
    }
}

/// One finish request as the worker sees it: its own cancellation, so a
/// timeout aborts this job and never a later capture, and its reply.
pub(crate) struct Job {
    cancel: CancelToken,
    reply: Sender<Reply>,
}

impl Job {
    fn start(&self) {
        let _ = self.reply.send(Reply::Started);
    }

    fn finish(self, result: Result<Transcribed, String>) {
        let _ = self.reply.send(Reply::Done(result));
    }

    fn finish_turns(self, result: Result<Vec<SpeakerTurn>, String>) {
        let _ = self.reply.send(Reply::Turns(result));
    }
}

enum Reply {
    /// The engine began this job; its deadline starts now.
    Started,
    Done(Result<Transcribed, String>),
    Turns(Result<Vec<SpeakerTurn>, String>),
}

#[derive(Debug)]
pub struct Transcribed {
    /// The full result: text, detected language, and any segment/word timings
    /// the model produced. A stream finish carries text only (the streaming
    /// API reports text diffs, not rows).
    pub transcript: Transcript,
    pub source: Source,
    /// Time from the finish request to the final text.
    pub finish_ms: u64,
}

impl Transcribed {
    /// The final text, trimmed.
    pub fn text(&self) -> &str {
        self.transcript.text.trim()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Stream,
    Offline,
    /// The stream could not be used; the reason is logged with the result.
    Fallback(String),
}

impl Source {
    pub fn label(&self) -> &str {
        match self {
            Self::Stream => "stream",
            Self::Offline => "offline",
            Self::Fallback(_) => "fallback",
        }
    }
}

/// The token of the job or stream the worker is running, so dropping the
/// engine aborts it. Work entered after the engine stops starts cancelled.
#[derive(Clone, Default)]
struct Running(Arc<Mutex<RunningState>>);

#[derive(Default)]
struct RunningState {
    stopping: bool,
    cancel: Option<CancelToken>,
}

impl Running {
    fn enter(&self, cancel: &CancelToken) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if state.stopping {
            cancel.cancel();
        }
        state.cancel = Some(cancel.clone());
    }

    fn stop(&self) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.stopping = true;
        if let Some(cancel) = &state.cancel {
            cancel.cancel();
        }
    }
}

pub struct Engine {
    /// The command sender behind a lock so the engine is `Sync` and one
    /// loaded model can be shared (`Arc<Engine>`) across recognition
    /// threads, like wxasr's `Arc<Model>`. Ordering is unchanged: one
    /// channel, and a session's commands all come from its own thread.
    commands: Mutex<Sender<Command>>,
    running: Running,
    /// Set when a job misses its deadline and cleared when the worker starts
    /// another: jobs queued behind a wedged run fail instead of waiting.
    stalled: Arc<AtomicBool>,
    /// Dictation-class requests sent so far. Background work submitted before
    /// a newer request yields to it.
    requests: Arc<AtomicU64>,
    /// Whether the loaded model can stream on its device.
    streams: Arc<AtomicBool>,
    /// Whether a diarize model is loaded alongside the ASR one.
    diarizes: Arc<AtomicBool>,
    /// The speaker-split strategy the loaded model's timings make right —
    /// what [`SpeakerSplit::Auto`] resolves to. Set once at start, from the
    /// model's capability, never mutated after.
    split: SpeakerSplit,
    /// Whether dictation streams publish their text as they go.
    live_text: bool,
    /// Whether the worker is running a dictation stream or finish.
    dictating: Arc<AtomicBool>,
    next_stream: AtomicU64,
    deadlines: Deadlines,
    thread: Option<JoinHandle<()>>,
}

/// A live stream for one capture. Feed it the capture's 16 kHz audio in order,
/// then either [`Engine::finish`] or drop it to cancel.
pub struct LiveStream {
    id: u64,
    cancel: CancelToken,
    fed: usize,
    finished: bool,
    commands: Sender<Command>,
    /// The stream's text so far, and the last version [`Self::take_partial`]
    /// returned.
    partials: Option<(Partials, u64)>,
}

impl LiveStream {
    /// Total samples handed to the stream so far.
    pub fn fed(&self) -> usize {
        self.fed
    }

    /// The model's text for the audio fed so far, when it changed since the
    /// last call. Always `None` with live text off. It is provisional: the
    /// finish may replace it entirely, so it is only ever shown, never pasted
    /// or kept.
    pub fn take_partial(&mut self) -> Option<Partial> {
        let (partials, seen) = self.partials.as_mut()?;
        partials.newer_than(seen)
    }

    pub fn feed(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        self.fed += samples.len();
        let _ = self.commands.send(Command::Feed {
            stream: self.id,
            samples: samples.to_vec(),
        });
    }
}

impl Drop for LiveStream {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.commands.send(Command::Cancel { stream: self.id });
        }
    }
}

/// A finish in flight. Waiting enforces the stream deadline from the caller's
/// side, so a stalled native stream cannot hold the paste forever, then bounds
/// the job once the engine starts it, so a wedged run fails instead.
pub struct PendingTranscript {
    reply: Receiver<Reply>,
    cancel: CancelToken,
    stream_cancel: Option<CancelToken>,
    stalled: Arc<AtomicBool>,
    /// How long the job may run once started: its allowance plus the clip's.
    job: Duration,
    stream_finish: Duration,
}

impl PendingTranscript {
    pub fn wait(self) -> Result<Transcribed, String> {
        let mut stream_deadline = self
            .stream_cancel
            .as_ref()
            .map(|_| Instant::now() + self.stream_finish);
        let mut job_deadline: Option<Instant> = None;
        loop {
            let now = Instant::now();
            let timeout = [stream_deadline, job_deadline]
                .into_iter()
                .flatten()
                .min()
                .map_or(QUEUE_POLL, |deadline| {
                    deadline.saturating_duration_since(now).min(QUEUE_POLL)
                });
            match self.reply.recv_timeout(timeout) {
                Ok(Reply::Done(result)) => return result,
                Ok(Reply::Started) => job_deadline = Some(Instant::now() + self.job),
                // A diarize job's reply never reaches a transcript waiter.
                Ok(Reply::Turns(_)) => return Err("transcription engine stopped".into()),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("transcription engine stopped".into());
                }
                Err(RecvTimeoutError::Timeout) => {
                    let now = Instant::now();
                    if let (Some(deadline), Some(stream)) = (stream_deadline, &self.stream_cancel)
                        && now >= deadline
                    {
                        tracing::warn!("live stream missed its finish deadline; aborting it");
                        stream.cancel();
                        stream_deadline = None;
                    }
                    if job_deadline.is_some_and(|deadline| now >= deadline) {
                        tracing::error!(
                            deadline_ms = self.job.as_millis() as u64,
                            "transcription missed its deadline; failing the job"
                        );
                        self.stalled.store(true, Ordering::Release);
                        self.abandon();
                        return Err("transcription timed out".into());
                    }
                    if job_deadline.is_none() && self.stalled.load(Ordering::Acquire) {
                        tracing::error!("transcription engine is stuck; failing a queued job");
                        self.abandon();
                        return Err("transcription engine is stuck".into());
                    }
                }
            }
        }
    }

    /// Aborts this job's run, or skips it if the engine has not reached it.
    fn abandon(&self) {
        self.cancel.cancel();
        if let Some(stream) = &self.stream_cancel {
            stream.cancel();
        }
    }
}

/// A diarize job in flight, with the same bounded waits as a transcript job:
/// the deadline starts when the engine begins it, so time queued behind other
/// work is not counted, and a wedged engine fails the job instead of hanging.
pub struct PendingTurns {
    reply: Receiver<Reply>,
    cancel: CancelToken,
    stalled: Arc<AtomicBool>,
    /// How long the job may run once started: its allowance plus the clip's.
    job: Duration,
}

impl PendingTurns {
    pub fn wait(self) -> Result<Vec<SpeakerTurn>, String> {
        let mut job_deadline: Option<Instant> = None;
        loop {
            let now = Instant::now();
            let timeout = job_deadline.map_or(QUEUE_POLL, |deadline| {
                deadline.saturating_duration_since(now).min(QUEUE_POLL)
            });
            match self.reply.recv_timeout(timeout) {
                Ok(Reply::Turns(result)) => return result,
                Ok(Reply::Started) => job_deadline = Some(Instant::now() + self.job),
                // A transcript job's reply never reaches a turns waiter.
                Ok(Reply::Done(_)) => return Err("transcription engine stopped".into()),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("transcription engine stopped".into());
                }
                Err(RecvTimeoutError::Timeout) => {
                    let now = Instant::now();
                    if job_deadline.is_some_and(|deadline| now >= deadline) {
                        tracing::error!(
                            deadline_ms = self.job.as_millis() as u64,
                            "diarization missed its deadline; failing the job"
                        );
                        self.stalled.store(true, Ordering::Release);
                        self.cancel.cancel();
                        return Err("diarization timed out".into());
                    }
                    if job_deadline.is_none() && self.stalled.load(Ordering::Acquire) {
                        tracing::error!("transcription engine is stuck; failing a queued job");
                        self.cancel.cancel();
                        return Err("transcription engine is stuck".into());
                    }
                }
            }
        }
    }
}

impl Engine {
    /// Loads `spec` on `device` and starts its dedicated engine thread,
    /// keeping Hex's queueing, priorities, deadlines, and bounded waits.
    pub fn load(spec: &ModelSpec, device: Device) -> Result<Self> {
        Self::start(Model::load(spec, device)?)
    }

    /// [`Self::load`] with an optional standalone diarize model loaded next
    /// to the ASR one. A diarize model that fails to load only warns: the
    /// engine transcribes without speaker attribution ([`Self::diarizes`]
    /// reports which way it went), never failing the ASR load.
    pub fn load_with_diarize(
        spec: &ModelSpec,
        device: Device,
        diarize: Option<&std::path::Path>,
    ) -> Result<Self> {
        let model = Model::load(spec, device)?;
        // A failed diarize load degrades to no diarization, never fails the engine.
        let diarizer = match diarize {
            Some(path) => Diarizer::load(path, device)
                .map(Some)
                .unwrap_or_else(|error| {
                    tracing::warn!(
                        error = format!("{error:#}"),
                        path = path.display().to_string(),
                        "could not load the diarize model; transcribing without speaker attribution"
                    );
                    None
                }),
            None => None,
        };
        Self::start_with(model, diarizer)
    }

    /// The engine's command sender, cloned under its lock.
    fn sender(&self) -> Sender<Command> {
        self.commands
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// Sends `command` to the worker, reporting whether it is still running.
    fn send(&self, command: Command) -> std::result::Result<(), mpsc::SendError<Command>> {
        self.commands
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .send(command)
    }

    /// Starts the engine for an already-loaded model. Wrap it in an `Arc` to
    /// share one loaded model across sessions and threads ([`Engine`] is
    /// `Send + Sync`).
    pub fn start(model: Model) -> Result<Self> {
        Self::start_with(model, None)
    }

    /// [`Self::start`] with an optional diarize model the worker owns next to
    /// the ASR one, so diarize runs serialize with transcriptions on the same
    /// thread instead of racing for the GPU.
    pub fn start_with(model: Model, diarizer: Option<Diarizer>) -> Result<Self> {
        let streams = model.stream_options().is_some();
        let streams = Arc::new(AtomicBool::new(streams));
        let diarizes = Arc::new(AtomicBool::new(diarizer.is_some()));
        // The runtime truth `SpeakerSplit::Auto` resolves to; the model is
        // moved into the worker below, so read its capability now. The
        // header-only guess for before a load is `gguf_meta` +
        // `speaker_split_for`.
        let split = if matches!(
            model.finest_timestamps(),
            transcribe_cpp::TimestampKind::None
        ) {
            SpeakerSplit::Pieces
        } else {
            SpeakerSplit::MixedSegments
        };
        let (commands, receiver) = mpsc::channel();
        let requests = Arc::new(AtomicU64::new(0));
        let running = Running::default();
        let stalled = Arc::new(AtomicBool::new(false));
        let dictating = Arc::new(AtomicBool::new(false));
        let worker = Worker {
            model,
            diarizer,
            running: running.clone(),
            stalled: stalled.clone(),
            streams: streams.clone(),
            requests: requests.clone(),
            dictating: dictating.clone(),
        };
        let thread = thread::Builder::new()
            .name("ggufasr-engine".into())
            .spawn(move || worker.run(receiver))?;
        Ok(Self {
            commands: Mutex::new(commands),
            running,
            stalled,
            requests,
            streams,
            diarizes,
            split,
            live_text: true,
            dictating,
            next_stream: AtomicU64::new(0),
            deadlines: Deadlines::PRODUCTION,
            thread: Some(thread),
        })
    }

    /// An engine without a worker whose model always streams; tests read the
    /// commands it would have handled.
    #[cfg(test)]
    pub(crate) fn for_test() -> (Self, Receiver<Command>) {
        let (commands, receiver) = mpsc::channel();
        let engine = Self {
            commands: Mutex::new(commands),
            running: Running::default(),
            stalled: Arc::default(),
            requests: Arc::default(),
            streams: Arc::new(AtomicBool::new(true)),
            diarizes: Arc::new(AtomicBool::new(false)),
            split: SpeakerSplit::MixedSegments,
            live_text: true,
            dictating: Arc::default(),
            next_stream: AtomicU64::new(0),
            deadlines: Deadlines::PRODUCTION,
            thread: None,
        };
        (engine, receiver)
    }

    /// Like [`Self::for_test`] for a model that cannot stream.
    #[cfg(test)]
    pub(crate) fn for_test_without_streaming() -> (Self, Receiver<Command>) {
        let (engine, receiver) = Self::for_test();
        engine.streams.store(false, Ordering::Release);
        (engine, receiver)
    }

    #[cfg(test)]
    fn with_deadlines(mut self, deadlines: Deadlines) -> Self {
        self.deadlines = deadlines;
        self
    }

    /// Turns live text off, so streams publish no partials.
    #[cfg(test)]
    fn without_live_text(mut self) -> Self {
        self.live_text = false;
        self
    }

    /// Pretends a diarize model is loaded, so tests can send its command.
    #[cfg(test)]
    fn with_test_diarize(self) -> Self {
        self.diarizes.store(true, Ordering::Release);
        self
    }

    /// Opens a live stream when the loaded model can stream on this device.
    pub fn begin(&self) -> Option<LiveStream> {
        self.begin_ignoring_head(0)
    }

    /// Opens a live stream whose opening-coverage check ignores `head_ignored`
    /// clip samples: the caller's own sounds at the capture's head, never a
    /// lost opening.
    pub fn begin_ignoring_head(&self, head_ignored: usize) -> Option<LiveStream> {
        if !self.streams.load(Ordering::Acquire) {
            return None;
        }
        let id = self.next_stream.fetch_add(1, Ordering::Relaxed) + 1;
        let cancel = CancelToken::new();
        let partials = self.live_text.then(Partials::default);
        self.requests.fetch_add(1, Ordering::AcqRel);
        self.send(Command::Begin {
            stream: id,
            cancel: cancel.clone(),
            partials: partials.clone(),
            head_ignored,
        })
        .ok()?;
        Some(LiveStream {
            id,
            cancel,
            fed: 0,
            finished: false,
            commands: self.sender(),
            partials: partials.map(|partials| (partials, 0)),
        })
    }

    /// Opens a stream that gives way to dictation, when the loaded model can
    /// stream on this device. See [`YieldingStream`].
    pub fn begin_yielding(&self) -> Option<YieldingStream> {
        if !self.streams.load(Ordering::Acquire) {
            return None;
        }
        let id = self.next_stream.fetch_add(1, Ordering::Relaxed) + 1;
        let cancel = CancelToken::new();
        let yielding = Yielding::new();
        self.send(Command::BeginYielding {
            stream: id,
            cancel: cancel.clone(),
            yielding: yielding.clone(),
        })
        .ok()?;
        Some(YieldingStream {
            id,
            cancel,
            yielding,
            sent: 0,
            finished: false,
            commands: self.sender(),
        })
    }

    /// Whether the loaded model can stream on its device.
    pub fn streams(&self) -> bool {
        self.streams.load(Ordering::Acquire)
    }

    /// Whether the engine is running a dictation stream or finish, so
    /// lower-priority work gives way to it.
    pub fn busy(&self) -> bool {
        self.dictating.load(Ordering::Acquire)
    }

    /// Finishes a capture with its complete 16 kHz clip. Without a stream,
    /// or when the stream is unusable, the clip is transcribed offline.
    pub fn finish(&self, stream: Option<LiveStream>, clip: Vec<f32>) -> Result<PendingTranscript> {
        // An engine without a worker thread (tests) has nobody to run the
        // job or drop the reply sender: waiting would poll forever. Tests
        // that drive jobs by hand use `finish_with` directly.
        if self.thread.is_none() {
            return Err(eyre!("transcription engine has no worker thread"));
        }
        self.finish_with(
            stream,
            clip,
            Box::new(|model, clip| {
                let clip = padded_whole_clip(clip, model.trailing_silence_samples());
                model
                    .transcribe(&clip)
                    .map_err(|error| format!("{error:#}"))
            }),
        )
    }

    /// Like `finish`, with the caller's offline transcription of the clip.
    pub fn finish_with(
        &self,
        stream: Option<LiveStream>,
        clip: Vec<f32>,
        offline: Offline,
    ) -> Result<PendingTranscript> {
        let (stream, stream_cancel) = match stream {
            Some(mut stream) => {
                stream.finished = true;
                (Some(stream.id), Some(stream.cancel.clone()))
            }
            None => (None, None),
        };
        self.request(clip.len(), stream_cancel, |job| Command::Finish {
            stream,
            clip,
            offline,
            job,
        })
    }

    /// Transcribes `clip` offline after every command already sent, without
    /// ever interrupting a live stream: one open when it arrives finishes
    /// first. Background work sent before it yields, as to a finish.
    pub fn foreground(&self, clip: Vec<f32>, offline: Offline) -> Result<PendingTranscript> {
        self.request(clip.len(), None, |job| Command::Foreground {
            clip,
            offline,
            job,
        })
    }

    fn request(
        &self,
        samples: usize,
        stream_cancel: Option<CancelToken>,
        command: impl FnOnce(Job) -> Command,
    ) -> Result<PendingTranscript> {
        let (reply, receiver) = mpsc::channel();
        let cancel = CancelToken::new();
        let job = self.deadlines.job_for(samples);
        self.requests.fetch_add(1, Ordering::AcqRel);
        self.send(command(Job {
            cancel: cancel.clone(),
            reply,
        }))
        .map_err(|_| eyre!("transcription engine stopped"))?;
        Ok(PendingTranscript {
            reply: receiver,
            cancel,
            stream_cancel,
            stalled: self.stalled.clone(),
            job,
            stream_finish: self.deadlines.stream_finish,
        })
    }

    /// Transcribes a complete 16 kHz clip once, offline, after every command
    /// already sent. This is `finish(None, clip)` without needing a stream.
    pub fn transcribe_clip(&self, clip: Vec<f32>) -> Result<PendingTranscript> {
        self.finish(None, clip)
    }

    /// Whether a diarize model is loaded, so speaker turns can be run at all.
    pub fn diarizes(&self) -> bool {
        self.diarizes.load(Ordering::Acquire)
    }

    /// The speaker-split strategy the loaded model's timings make right —
    /// what [`SpeakerSplit::Auto`] resolves to, here from the model's real
    /// capability. The header-only guess for before a load is
    /// `gguf_meta` + [`crate::speaker_split_for`].
    pub fn speaker_split(&self) -> SpeakerSplit {
        self.split
    }

    /// Diarizes a complete 16 kHz clip on the engine's diarize model, after
    /// every command already sent. The turns are relative to the clip's
    /// start, and ids are only self-consistent within this one run. Fails at
    /// once when no diarize model is loaded (see [`Self::diarizes`]).
    pub fn diarize(&self, pcm: &[f32]) -> Result<PendingTurns> {
        if !self.diarizes() {
            return Err(eyre!("no diarize model is loaded"));
        }
        if self.thread.is_none() {
            return Err(eyre!("transcription engine has no worker thread"));
        }
        self.diarize_job(pcm)
    }

    /// [`Self::diarize`] without the worker check, so tests drive the command
    /// by hand, as they do `finish_with`.
    fn diarize_job(&self, pcm: &[f32]) -> Result<PendingTurns> {
        let (reply, receiver) = mpsc::channel();
        let cancel = CancelToken::new();
        let job = self.deadlines.job_for(pcm.len());
        self.send(Command::Diarize {
            pcm: pcm.to_vec(),
            job: Job {
                cancel: cancel.clone(),
                reply,
            },
        })
        .map_err(|_| eyre!("transcription engine stopped"))?;
        Ok(PendingTurns {
            reply: receiver,
            cancel,
            stalled: self.stalled.clone(),
            job,
        })
    }

    /// Transcribes a complete clip with segment and word timings, then merges
    /// diarize `turns` into its segments: the authoritative speaker-attributed
    /// text of one rotation segment. `turns` must be on `pcm`'s timeline (a
    /// rolling-window run needs its offset shifted to the segment first).
    /// `split` picks the speaker-split strategy — `Auto` resolves against the
    /// loaded model ([`Self::speaker_split`]) once, here: a model without any
    /// timing answers one segment per clip however many people spoke, so its
    /// clip is cut where the turns say the speaker changed and transcribed
    /// piecewise; a timed model keeps its own segments, re-splitting only
    /// those that straddle a speaker change. A failed piece falls back to the
    /// whole-clip run and its whole-segment attribution.
    pub fn transcribe_clip_turns(
        &self,
        pcm: Vec<f32>,
        turns: Vec<SpeakerTurn>,
        split: SpeakerSplit,
    ) -> Result<PendingTranscript> {
        // Resolve `Auto` against the loaded model now, so a clip's strategy
        // does not depend on when the engine picks it up.
        let split = if split == SpeakerSplit::Auto {
            self.speaker_split()
        } else {
            split
        };
        self.finish_with(
            None,
            pcm,
            Box::new(move |model, clip| {
                match split {
                    SpeakerSplit::Pieces => {
                        match crate::diarize::split_transcript(&clip, &turns, |piece| {
                            model.transcribe_timed(piece.to_vec())
                        }) {
                            Ok(transcript) => return Ok(transcript),
                            Err(error) => tracing::warn!(
                                error = format!("{error:#}"),
                                "piecewise transcription failed; attributing the whole clip"
                            ),
                        }
                    }
                    SpeakerSplit::MixedSegments => {
                        // The re-split needs the audio after the timed run;
                        // only this branch pays the copy.
                        let mut transcript = model
                            .transcribe_timed(clip.clone())
                            .map_err(|error| format!("{error:#}"))?;
                        crate::diarize::assign_speakers(&mut transcript, &turns);
                        let resplit = crate::diarize::resplit_mixed_segments(
                            &clip,
                            &mut transcript,
                            &turns,
                            |piece| model.transcribe_timed(piece.to_vec()),
                        );
                        if resplit > 0 {
                            tracing::debug!(
                                resplit,
                                "re-split segments straddling a speaker change"
                            );
                        }
                        return Ok(transcript);
                    }
                    // `Off`, and `Auto` is resolved away above.
                    SpeakerSplit::Auto | SpeakerSplit::Off => {}
                }
                let mut transcript = model
                    .transcribe_timed(clip)
                    .map_err(|error| format!("{error:#}"))?;
                crate::diarize::assign_speakers(&mut transcript, &turns);
                Ok(transcript)
            }),
        )
    }

    /// Runs `work` with the model unless a dictation request was sent first or
    /// a live stream is open, and returns `None` when one was.
    pub fn background<R: Send + 'static>(
        &self,
        work: impl FnOnce(&mut Model) -> R + Send + 'static,
    ) -> Result<Option<R>> {
        if self.thread.is_none() {
            return Err(eyre!("transcription engine has no worker thread"));
        }
        let (reply, receiver) = mpsc::channel();
        let generation = self.requests.load(Ordering::Acquire);
        self.send(Command::Background {
            generation,
            work: Box::new(move |model| {
                let _ = reply.send(model.map(work));
            }),
        })
        .map_err(|_| eyre!("transcription engine stopped"))?;
        receiver
            .recv()
            .map_err(|_| eyre!("transcription engine stopped"))
    }

    /// Drops the engine on its own thread, so replacing the model never waits
    /// on a run still in flight.
    pub fn retire(self) {
        let spawned = thread::Builder::new()
            .name("ggufasr-engine-retire".into())
            .spawn(move || drop(self));
        if let Err(error) = spawned {
            tracing::warn!(%error, "could not retire the old transcription engine in the background");
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.running.stop();
        // A live stream's sender may outlive the engine; stop explicitly.
        let _ = self.send(Command::Shutdown);
        let Some(thread) = self.thread.take() else {
            return;
        };
        let deadline = Instant::now() + self.deadlines.shutdown;
        while !thread.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        if thread.is_finished() {
            let _ = thread.join();
        } else {
            tracing::error!("transcription engine did not stop in time; leaving it running");
        }
    }
}

/// A whole clip as a dictation run transcribes it: the minimum duration and
/// the model's trailing silence, so a trailing pause cannot trim Unified
/// English's closing context (#1629). Long pauses are never cut here (Hex's
/// Labs › Skip long pauses is off by default and stays out of this build).
fn padded_whole_clip(clip: Vec<f32>, trailing_silence: usize) -> Vec<f32> {
    let mut clip = whole_clip(clip, SKIP_LONG_PAUSES);
    crate::pad_gguf_samples(&mut clip, trailing_silence);
    clip
}

/// Hex's Labs gate, inlined: this build never cuts a whole clip's pauses.
pub(crate) const SKIP_LONG_PAUSES: bool = false;

struct Worker {
    model: Model,
    /// The diarize model, when one was loaded; run only on this thread, so
    /// it never races an ASR run for the GPU.
    diarizer: Option<Diarizer>,
    running: Running,
    stalled: Arc<AtomicBool>,
    streams: Arc<AtomicBool>,
    requests: Arc<AtomicU64>,
    dictating: Arc<AtomicBool>,
}

/// How a live stream ended inside the worker.
enum StreamEnd {
    /// A finish for this stream: the final text, or why the stream is unusable.
    Finished {
        result: Result<String, String>,
        clip: Vec<f32>,
        offline: Offline,
        job: Job,
        started: Instant,
    },
    Cancelled,
    /// Another command arrived first; it is handled after the stream closes.
    Interrupted(Command),
    Disconnected,
}

impl Worker {
    fn run(mut self, commands: Receiver<Command>) {
        // Commands received while a stream was open, handled before new ones.
        let mut next = VecDeque::new();
        loop {
            let Some(command) = next.pop_front().or_else(|| commands.recv().ok()) else {
                return;
            };
            match command {
                Command::Begin {
                    stream,
                    cancel,
                    partials,
                    head_ignored,
                } => {
                    self.dictating.store(true, Ordering::Release);
                    self.stream(
                        stream,
                        &cancel,
                        partials.as_ref(),
                        head_ignored,
                        &commands,
                        &mut next,
                    );
                    self.dictating.store(false, Ordering::Release);
                }
                Command::BeginYielding {
                    stream,
                    cancel,
                    yielding,
                } => {
                    // Anything already waiting comes first.
                    if next.is_empty() {
                        self.yielding_stream(stream, &cancel, &yielding, &commands, &mut next);
                    }
                    yielding.close();
                }
                Command::FinishYielding { reply, .. } => {
                    let _ = reply.send(Err(PREEMPTED.into()));
                }
                Command::Finish {
                    stream,
                    clip,
                    offline,
                    job,
                } => {
                    let source = match stream {
                        // The stream was already closed: its begin failed or
                        // another command interrupted it.
                        Some(_) => Source::Fallback("the live stream was not available".into()),
                        None => Source::Offline,
                    };
                    self.dictating.store(true, Ordering::Release);
                    self.run_offline(clip, offline, source, job);
                    self.dictating.store(false, Ordering::Release);
                }
                Command::Foreground { clip, offline, job } => {
                    self.run_offline(clip, offline, Source::Offline, job);
                }
                Command::Background { generation, work } => {
                    let current = self.requests.load(Ordering::Acquire) == generation;
                    work(if current { Some(&mut self.model) } else { None });
                }
                Command::Diarize { pcm, job } => {
                    self.run_diarize(pcm, job);
                }
                Command::Shutdown => return,
                // Stale input for a stream that already ended.
                Command::Feed { .. } | Command::Cancel { .. } => {}
            }
            self.publish_streams();
        }
    }

    /// Whether the model can stream on its device.
    fn publish_streams(&self) {
        self.streams
            .store(self.model.stream_options().is_some(), Ordering::Release);
    }

    fn run_offline(&mut self, clip: Vec<f32>, offline: Offline, source: Source, job: Job) {
        if job.cancel.is_cancelled() {
            // Its waiter already gave up on it.
            job.finish(Err("transcription was abandoned".into()));
            return;
        }
        let started = Instant::now();
        self.start(&job);
        self.offline(clip, offline, source, started, job);
    }

    /// Starts a job's clock. The engine is evidently moving again.
    fn start(&self, job: &Job) {
        self.stalled.store(false, Ordering::Release);
        job.start();
    }

    /// Drives one live stream until it finishes, is cancelled, or another
    /// command interrupts it. Commands still to be handled, in order, are
    /// pushed onto `next`.
    fn stream(
        &mut self,
        id: u64,
        cancel: &CancelToken,
        partials: Option<&Partials>,
        head_ignored: usize,
        commands: &Receiver<Command>,
        next: &mut VecDeque<Command>,
    ) {
        let Some(options) = self.model.stream_options().cloned() else {
            tracing::info!(
                stream = id,
                live_text = partials.is_some(),
                "a live stream was requested but this model can't stream on its device; transcribing the whole clip"
            );
            return;
        };
        self.running.enter(cancel);
        let begun = Instant::now();
        let mut stats = FeedStats::default();
        let stalled = &self.stalled;
        let trailing_silence = stream_trailing_silence(&self.model);
        let ended = self.model.with_stream(&options, cancel, |stream| {
            let mut capped = PauseCap::new(stream, trailing_silence).ignoring_head(head_ignored);
            let ended = drive_stream(
                id,
                &mut capped,
                commands,
                stalled,
                partials,
                next,
                &mut stats,
            );
            stats.pauses_dropped_ms = capped.dropped / 16;
            ended
        });
        let FeedStats {
            feed_ms,
            feeds,
            partials: changes,
            pauses_dropped_ms,
        } = stats;
        match ended {
            Ok(StreamEnd::Finished {
                result,
                clip,
                offline,
                job,
                started,
            }) => {
                tracing::info!(
                    audio_ms = clip.len() / 16,
                    stream_ms = begun.elapsed().as_millis(),
                    feeds,
                    feed_ms,
                    partials = changes,
                    pauses_dropped_ms,
                    finish_ms = started.elapsed().as_millis(),
                    "finished live stream"
                );
                match result {
                    Ok(text) => job.finish(Ok(Transcribed {
                        transcript: Transcript {
                            text,
                            ..Transcript::default()
                        },
                        source: Source::Stream,
                        finish_ms: started.elapsed().as_millis() as u64,
                    })),
                    Err(_) if job.cancel.is_cancelled() => {
                        job.finish(Err("transcription was abandoned".into()));
                    }
                    Err(reason) => {
                        tracing::warn!(%reason, "live stream unusable; transcribing the full clip");
                        self.offline(clip, offline, Source::Fallback(reason), started, job);
                    }
                }
            }
            Ok(StreamEnd::Cancelled) | Ok(StreamEnd::Disconnected) => {}
            Ok(StreamEnd::Interrupted(command)) => {
                // The capture goes on without its stream (a model switch):
                // its text stops, so the plate hides.
                if let Some(partials) = partials {
                    partials.publish(Partial::default());
                }
                next.push_back(command);
            }
            Err(error) => tracing::warn!(%error, "could not begin a live stream"),
        }
    }

    /// Drives one yielding stream until it finishes, is cancelled, or any
    /// command that is not its own, background, or foreground work arrives;
    /// that command is handled next.
    fn yielding_stream(
        &mut self,
        id: u64,
        cancel: &CancelToken,
        yielding: &Yielding,
        commands: &Receiver<Command>,
        next: &mut VecDeque<Command>,
    ) {
        let Some(options) = self.model.stream_options().cloned() else {
            return;
        };
        if cancel.is_cancelled() {
            return;
        }
        self.running.enter(cancel);
        let ended = self.model.with_stream(&options, cancel, |stream| {
            drive_yielding(id, stream, commands, yielding, next)
        });
        match ended {
            Ok(Some(command)) => next.push_back(command),
            Ok(None) => {}
            Err(error) => tracing::warn!(%error, "could not begin a yielding stream"),
        }
    }

    fn offline(
        &mut self,
        clip: Vec<f32>,
        offline: Offline,
        source: Source,
        started: Instant,
        job: Job,
    ) {
        self.running.enter(&job.cancel);
        let result =
            crate::gguf_session::with_job_cancel(&job.cancel, || offline(&mut self.model, clip))
                .map(|mut transcript| {
                    transcript.text = transcript.text.trim().to_string();
                    Transcribed {
                        transcript,
                        source,
                        finish_ms: started.elapsed().as_millis() as u64,
                    }
                });
        job.finish(result);
    }

    /// One diarize run, as a job with its own deadline and cancellation.
    /// Without a diarize model it answers at once: the caller degrades.
    fn run_diarize(&mut self, pcm: Vec<f32>, job: Job) {
        if job.cancel.is_cancelled() {
            job.finish_turns(Err("diarization was abandoned".into()));
            return;
        }
        // Start the clock and enter the run token before borrowing the
        // diarizer: locks and borrows taken once, never re-taken while held
        // (the scrutinee-guard deadlock this codebase already met once).
        self.start(&job);
        self.running.enter(&job.cancel);
        let result = match self.diarizer.as_mut() {
            Some(diarizer) => {
                crate::gguf_session::with_job_cancel(&job.cancel, || diarizer.run(&pcm))
                    .map_err(|error| format!("{error:#}"))
            }
            None => Err("no diarize model is loaded".into()),
        };
        job.finish_turns(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_labels_distinguish_fallbacks() {
        assert_eq!(Source::Stream.label(), "stream");
        assert_eq!(Source::Offline.label(), "offline");
        assert_eq!(Source::Fallback("x".into()).label(), "fallback");
    }

    #[test]
    fn streams_send_their_audio_in_order_and_cancel_when_dropped() {
        let (engine, commands) = Engine::for_test();
        let mut stream = engine.begin().expect("the test engine streams");
        stream.feed(&[0.1; 3]);
        stream.feed(&[]);
        stream.feed(&[0.2; 2]);
        assert_eq!(stream.fed(), 5);
        drop(stream);
        let finished = engine.begin().expect("the test engine streams");
        let pending = engine
            .finish_with(
                Some(finished),
                vec![0.0; 7],
                Box::new(|_, _| Ok(Transcript::default())),
            )
            .unwrap();
        assert!(pending.stream_cancel.is_some());
        let received = commands.try_iter().collect::<Vec<_>>();
        assert!(matches!(
            received.as_slice(),
            [
                Command::Begin { stream: 1, .. },
                Command::Feed { stream: 1, samples: first },
                Command::Feed { stream: 1, samples: second },
                Command::Cancel { stream: 1 },
                Command::Begin { stream: 2, .. },
                Command::Finish { stream: Some(2), clip, .. },
            ] if first.len() == 3 && second.len() == 2 && clip.len() == 7
        ));
    }

    #[derive(Default)]
    struct FakeStream {
        fed: Vec<f32>,
        /// Fail every feed once this many samples are in, as a native
        /// stream error would.
        fail_after: Option<usize>,
        /// Where its finalized text starts, in milliseconds.
        text_start_ms: Option<u64>,
        /// Where its finalized text ends, in milliseconds.
        text_end_ms: Option<u64>,
        /// Language tags its raw decode carries.
        tags: Vec<String>,
    }

    impl StreamFeed for FakeStream {
        fn feed(&mut self, samples: &[f32]) -> Result<(), String> {
            if self.fail_after.is_some_and(|after| self.fed.len() >= after) {
                return Err("native feed failed".into());
            }
            self.fed.extend_from_slice(samples);
            Ok(())
        }

        fn finalize(&mut self) -> Result<String, String> {
            Ok(format!("streamed {}", self.fed.len()))
        }

        fn partial(&self) -> Partial {
            Partial {
                stable: format!("heard {}", self.fed.len()),
                tentative: String::new(),
            }
        }

        fn text_start_ms(&self) -> Option<u64> {
            self.text_start_ms
        }

        fn text_end_ms(&self) -> Option<u64> {
            self.text_end_ms
        }

        fn language_tags(&self) -> Vec<String> {
            self.tags.clone()
        }
    }

    #[test]
    fn foreground_work_waits_for_an_open_stream_instead_of_interrupting_it() {
        let (engine, commands) = Engine::for_test();
        let mut stream = engine.begin().unwrap();
        assert!(matches!(
            commands.try_recv().unwrap(),
            Command::Begin { stream: 1, .. }
        ));
        stream.feed(&[0.1; 4]);
        let chunk = engine
            .foreground(vec![0.0; 9], Box::new(|_, _| Ok(text_transcript("chunk"))))
            .unwrap();
        stream.feed(&[0.2; 4]);
        let dictation = engine
            .finish_with(
                Some(stream),
                vec![0.0; 8],
                Box::new(|_, _| panic!("the dictation must not be retranscribed offline")),
            )
            .unwrap();

        let mut fake = FakeStream::default();
        let mut deferred = VecDeque::new();
        let ended = drive_stream(
            1,
            &mut fake,
            &commands,
            &AtomicBool::new(false),
            None,
            &mut deferred,
            &mut FeedStats::default(),
        );
        let StreamEnd::Finished { result, job, .. } = ended else {
            panic!("the stream did not finish");
        };
        // The stream got all its audio and finished as a stream.
        assert_eq!(result, Ok("streamed 8".to_string()));
        assert_eq!(fake.fed.len(), 8);
        job.finish(Ok(Transcribed {
            transcript: text_transcript("streamed 8"),
            source: Source::Stream,
            finish_ms: 0,
        }));
        assert_eq!(dictation.wait().unwrap().source, Source::Stream);
        // The chunk waits to run after the stream.
        assert!(matches!(
            deferred.as_slices().0,
            [Command::Foreground { clip, .. }] if clip.len() == 9
        ));
        drop(deferred);
        assert!(chunk.wait().is_err());
    }

    fn tone(samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|i| 0.3 * (i as f32 * 0.05).sin())
            .collect()
    }

    #[test]
    fn a_dictation_stream_hears_only_the_ends_of_long_pauses() {
        // Quiet first, as the pre-roll is, so the noise floor starts low.
        let mut clip = vec![0.0; 8_000];
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 16_000]);
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 160_000]);
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 80_000]);
        let mut fake = FakeStream::default();
        let mut capped = PauseCap::new(&mut fake, 0);
        for chunk in clip.chunks(1_234) {
            capped.feed(chunk).unwrap();
        }
        capped.finalize().unwrap();
        let dropped = capped.dropped;
        // A 1 s pause passes whole; the 10 s one keeps 2.5 s and 0.3 s, and
        // the trailing 5 s one only its head.
        let long = 160_000 - PAUSE_HEAD_SAMPLES - PAUSE_TAIL_SAMPLES;
        let trailing = 80_000 - PAUSE_HEAD_SAMPLES;
        assert_eq!(dropped, long + trailing);
        assert_eq!(fake.fed.len(), clip.len() - dropped);
        assert_eq!(fake.fed[..56_000], clip[..56_000]);
        let resumed = 56_000 + PAUSE_HEAD_SAMPLES + PAUSE_TAIL_SAMPLES;
        assert_eq!(fake.fed[resumed..resumed + 16_000], clip[216_000..232_000]);
    }

    #[test]
    fn quiet_speech_over_steady_noise_is_never_cut_as_a_pause() {
        // #458: -28 dBFS syllables over -35 dBFS noise, too close to the
        // floor for the meeting detector's 12 dB speech margin.
        assert_quiet_speech_is_kept(-28.0, -35.0, 0.0);
        // #841: a quiet built-in microphone, whose speech sits under
        // -55 dBFS throughout.
        assert_quiet_speech_is_kept(-62.0, -72.0, 0.0);
        // #841: a built-in digital microphone whose DC offset (-26 dBFS) is
        // far louder than its speech; only the speech moves the level.
        assert_quiet_speech_is_kept(-50.0, -58.0, 0.05);
    }

    #[test]
    fn a_stream_that_heard_only_quiet_audio_falls_back_to_the_whole_clip() {
        use crate::pause_segmenter::tests::noise;
        // #841: the cap took a whole dictation for one pause and the stream
        // kept only its first 2.5 s; that stream must not be used.
        let mut seed = 3;
        let clip = noise(6_000, -80.0, &mut seed);
        let mut fake = FakeStream::default();
        let mut capped = PauseCap::new(&mut fake, 0);
        let result = finish_stream(&mut capped, &clip, 0);
        assert!(capped.dropped > 0);
        assert!(result.is_err(), "{result:?}");
    }

    /// Quiet pre-roll, 2 s of speech, a 1 s pause, `phrase` samples of
    /// speech, then the 200 ms tail after the stop.
    fn speech_pause_phrase(phrase: usize) -> Vec<f32> {
        let mut clip = vec![0.0; 8_000];
        clip.extend(tone(32_000));
        clip.extend(vec![0.0; 16_000]);
        clip.extend(tone(phrase));
        clip.extend(vec![0.0; 3_200]);
        clip
    }

    fn finish_with_text_end(clip: &[f32], text_end_ms: u64) -> Result<String, String> {
        finish_with_text_bounds(clip, None, text_end_ms)
    }

    fn finish_with_text_bounds(
        clip: &[f32],
        text_start_ms: Option<u64>,
        text_end_ms: u64,
    ) -> Result<String, String> {
        finish_with_text_bounds_ignoring_head(clip, 0, text_start_ms, text_end_ms)
    }

    /// As above, with `head_ignored` clip samples of the caller's own sounds
    /// at the capture's head.
    fn finish_with_text_bounds_ignoring_head(
        clip: &[f32],
        head_ignored: usize,
        text_start_ms: Option<u64>,
        text_end_ms: u64,
    ) -> Result<String, String> {
        let mut fake = FakeStream {
            text_start_ms,
            text_end_ms: Some(text_end_ms),
            ..FakeStream::default()
        };
        let mut capped = PauseCap::new(&mut fake, 0).ignoring_head(head_ignored);
        for chunk in clip.chunks(1_600) {
            capped.feed(chunk).unwrap();
        }
        let result = finish_stream(&mut capped, clip, clip.len());
        assert_eq!(fake.fed, clip);
        result
    }

    fn finish_with_tags(clip: &[f32], tags: Vec<String>) -> Result<String, String> {
        let mut fake = FakeStream {
            text_start_ms: Some(500),
            text_end_ms: Some(4_200),
            tags,
            ..FakeStream::default()
        };
        let mut capped = PauseCap::new(&mut fake, 0);
        for chunk in clip.chunks(1_600) {
            capped.feed(chunk).unwrap();
        }
        let result = finish_stream(&mut capped, clip, clip.len());
        assert_eq!(fake.fed, clip);
        result
    }

    #[test]
    fn a_stream_that_dropped_a_phrase_after_a_pause_falls_back_to_the_whole_clip() {
        // #1101: speech, a 1 s pause, a short last phrase, then stop. The
        // stream heard it all, but its text ended with the first speech.
        let clip = speech_pause_phrase(12_800);
        let result = finish_with_text_end(&clip, 2_500);
        assert!(
            result
                .as_ref()
                .is_err_and(|reason| reason.contains("speech follows")),
            "{result:?}"
        );
        // Text that reaches the phrase is the stream's to keep.
        assert_eq!(
            finish_with_text_end(&clip, 4_200),
            Ok(format!("streamed {}", clip.len()))
        );
    }

    #[test]
    fn a_closing_full_stop_at_the_clips_end_does_not_cover_a_dropped_phrase() {
        // Nemotron 3.5 ASR puts its final full stop on the clip's last frame
        // (apollo, FLEURS: the last word ended at 8.8 s, the stop at 10.16 s).
        let token = |text: &str, t1_ms| transcribe_cpp::Token {
            text: text.into(),
            t1_ms,
            ..Default::default()
        };
        let tokens = [
            token(" channels", 2_480),
            token(",", 2_500),
            token(".", 4_400),
            token(" ", 4_400),
        ];
        assert_eq!(spoken_text_end_ms(&tokens), Some(2_480));
        assert_eq!(spoken_text_end_ms(&[token(".", 4_400)]), None);
    }

    #[test]
    fn a_short_sound_after_the_text_does_not_discard_the_stream() {
        // A key click or the stop tone in the tail is not a dropped phrase.
        let clip = speech_pause_phrase(2_400);
        assert_eq!(
            finish_with_text_end(&clip, 2_500),
            Ok(format!("streamed {}", clip.len()))
        );
    }

    #[test]
    fn a_stream_that_lost_the_opening_phrase_falls_back_to_the_whole_clip() {
        // Speech opens at 500 ms; text that starts at 3 s lost its opening.
        let clip = speech_pause_phrase(12_800);
        let result = finish_with_text_bounds(&clip, Some(3_000), 4_200);
        assert!(
            result
                .as_ref()
                .is_err_and(|reason| reason.contains("precedes")),
            "{result:?}"
        );
        // Text that starts with the speech is the stream's to keep.
        assert_eq!(
            finish_with_text_bounds(&clip, Some(500), 4_200),
            Ok(format!("streamed {}", clip.len()))
        );
    }

    #[test]
    fn the_callers_own_sounds_at_the_captures_head_are_not_a_lost_opening() {
        // A 450 ms pre-roll, a 272 ms start tone, the tap's click, and room
        // before the first word at 1.6 s: the stream lost nothing.
        let mut clip = vec![0.0; 7_200]; // 450 ms of pre-roll
        clip.extend(tone(4_352)); // the 272 ms start tone
        clip.extend(tone(1_600)); // the tap's click, 100 ms
        clip.extend(vec![0.0; 12_448]); // room up to 1.6 s
        clip.extend(tone(16_000)); // the dictation itself, 1.6-2.6 s
        clip.extend(vec![0.0; 3_200]);
        let head = 14_400; // 900 ms: the pre-roll, the tone, and its margin
        assert_eq!(
            finish_with_text_bounds_ignoring_head(&clip, head, Some(1_600), 2_600),
            Ok(format!("streamed {}", clip.len()))
        );
        // Without the exclusion the same sounds are 400 ms of "speech" before
        // the text, which is what a real model sees when they are all the head
        // holds.
        assert!(
            finish_with_text_bounds(&clip, Some(1_600), 2_600)
                .is_err_and(|reason| reason.contains("precedes")),
            "the head exclusion is what keeps this stream"
        );
    }

    #[test]
    fn a_lost_opening_after_a_long_pause_still_falls_back_to_the_whole_clip() {
        // The review's case: 1.6 s of speech the stream lost, then a 1 s
        // pause, then the text at 3.5 s. The pause is longer than the check's
        // 400 ms slack, so nothing but counting the speech itself can catch it.
        let mut clip = vec![0.0; 14_400]; // 900 ms: pre-roll and start tone
        clip.extend(tone(25_600)); // 1.6 s of lost speech, 0.9-2.5 s
        clip.extend(vec![0.0; 16_000]); // a 1 s pause
        clip.extend(tone(16_000)); // the text's own speech, 3.5-4.5 s
        clip.extend(vec![0.0; 3_200]);
        let result = finish_with_text_bounds_ignoring_head(&clip, 14_400, Some(3_500), 4_500);
        assert!(
            result.as_ref().is_err_and(
                |reason| reason.contains("precedes") && reason.contains("from 900 to 2500 ms")
            ),
            "{result:?}"
        );
    }

    #[test]
    fn a_short_click_before_the_text_does_not_discard_the_stream() {
        // A 150 ms click before the speech is not a lost opening.
        let mut clip = vec![0.0; 8_000];
        clip.extend(tone(2_400));
        clip.extend(vec![0.0; 16_000]);
        clip.extend(tone(12_800));
        clip.extend(vec![0.0; 3_200]);
        assert_eq!(
            finish_with_text_bounds(&clip, Some(1_600), 2_500),
            Ok(format!("streamed {}", clip.len()))
        );
    }

    #[test]
    fn a_punctuation_token_does_not_start_the_streams_text() {
        let token = |text: &str, t0_ms| transcribe_cpp::Token {
            text: text.into(),
            t0_ms,
            ..Default::default()
        };
        let tokens = [
            token(".", 40),
            token(" channels", 2_480),
            token(" later", 3_000),
        ];
        assert_eq!(spoken_text_start_ms(&tokens), Some(2_480));
        assert_eq!(spoken_text_start_ms(&[token(".", 40)]), None);
    }

    #[test]
    fn a_stream_that_switched_language_falls_back_to_the_whole_clip() {
        assert_eq!(language_tags("<en-US> hello <en-US> there"), ["<en-US>"]);
        assert_eq!(
            language_tags("<en-US> hello <de-DE>"),
            ["<en-US>", "<de-DE>"]
        );
        let clip = speech_pause_phrase(12_800);
        // One tag repeated, as one per segment, is not a switch.
        let tags = |raw: &str| {
            language_tags(raw)
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        let same = finish_with_tags(&clip, tags("<en-US> hello <en-US>"));
        assert_eq!(same, Ok(format!("streamed {}", clip.len())));
        let switched = finish_with_tags(&clip, tags("<en-US> hello <de-DE>"));
        assert!(
            switched
                .as_ref()
                .is_err_and(|reason| reason.contains("switched language from <en-US> to <de-DE>")),
            "{switched:?}"
        );
    }

    #[test]
    fn skipping_pauses_cuts_a_whole_clips_long_pauses_to_their_ends() {
        // #899: quiet pre-roll, speech, a 1 s pause, speech, a 10 s pause,
        // speech, and a 5 s trailing pause.
        let mut clip = vec![0.0; 8_000];
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 16_000]);
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 160_000]);
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 80_000]);
        let (kept, dropped) = skip_long_pauses(&clip).expect("long pauses are cut");
        let long = 160_000 - PAUSE_HEAD_SAMPLES - PAUSE_TAIL_SAMPLES;
        let trailing = 80_000 - PAUSE_HEAD_SAMPLES;
        assert_eq!(dropped, long + trailing);
        assert_eq!(kept.len(), clip.len() - dropped);
        // The short pause passes whole, and the speech after the long one
        // follows its head and tail.
        assert_eq!(kept[..56_000], clip[..56_000]);
        let resumed = 56_000 + PAUSE_HEAD_SAMPLES + PAUSE_TAIL_SAMPLES;
        assert_eq!(kept[resumed..resumed + 16_000], clip[216_000..232_000]);
        // The trailing pause keeps its head.
        assert_eq!(kept.len(), resumed + 16_000 + PAUSE_HEAD_SAMPLES);
    }

    #[test]
    fn skipping_pauses_leaves_short_pauses_and_quiet_only_clips_whole() {
        use crate::pause_segmenter::tests::noise;
        let mut clip = vec![0.0; 8_000];
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 16_000]);
        clip.extend(tone(16_000));
        assert!(skip_long_pauses(&clip).is_none());
        assert_eq!(whole_clip(clip.clone(), true), clip);
        // #841: nothing above the floor, so nothing is taken for a pause.
        let mut seed = 3;
        let quiet = noise(6_000, -80.0, &mut seed);
        assert!(skip_long_pauses(&quiet).is_none());
        assert_eq!(whole_clip(quiet.clone(), true), quiet);
    }

    #[test]
    fn with_skip_pauses_off_the_whole_clip_is_untouched() {
        let mut clip = vec![0.0; 8_000];
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 160_000]);
        clip.extend(tone(16_000));
        let untouched = whole_clip(clip.clone(), false);
        assert_eq!(
            untouched.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            clip.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
        );
        assert!(whole_clip(clip.clone(), true).len() < clip.len());
    }

    #[test]
    fn a_whole_clip_is_padded_after_its_trailing_pause() {
        use crate::UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES;

        // #1629: quiet pre-roll, speech, then a 5 s trailing pause. This
        // build never cuts a whole clip's pauses (SKIP_LONG_PAUSES is false,
        // Hex's off-by-default Labs gate), so the clip runs whole and the
        // 400 ms of Unified English silence follows it, where nothing can
        // trim it.
        let mut clip = vec![0.0; 8_000];
        clip.extend(tone(16_000));
        clip.extend(vec![0.0; 80_000]);
        let cut = whole_clip(clip.clone(), true);
        let cut_len = cut.len();
        assert_eq!(cut_len, 8_000 + 16_000 + PAUSE_HEAD_SAMPLES);

        let padded = padded_whole_clip(clip.clone(), UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES);

        assert_eq!(
            padded.len(),
            clip.len() + UNIFIED_ENGLISH_TRAILING_SILENCE_SAMPLES
        );
        assert_eq!(padded[..clip.len()], clip[..]);
        assert!(padded[clip.len()..].iter().all(|&sample| sample == 0.0));
        // The cut a Labs build would make is a prefix of it.
        assert_eq!(padded[..cut_len], cut[..]);
    }

    fn assert_quiet_speech_is_kept(speech_db: f32, noise_db: f32, dc_offset: f32) {
        use crate::pause_segmenter::{FRAME_SAMPLES, SpeechFrames, tests::noise};
        // 17 s of syllables over steady noise.
        let mut seed = 11;
        let mut clip = noise(17_000, noise_db, &mut seed);
        let mut voice_seed = 5;
        for start in (0..clip.len()).step_by(4_800) {
            let burst = noise(240, speech_db, &mut voice_seed);
            for (sample, voice) in clip[start..].iter_mut().zip(burst) {
                *sample += voice;
            }
        }
        for sample in &mut clip {
            *sample += dc_offset;
        }
        let mut speech = SpeechFrames::default();
        assert!(
            !clip
                .as_chunks::<FRAME_SAMPLES>()
                .0
                .iter()
                .any(|frame| speech.is_speech(frame)),
            "the speech margin misses this speech"
        );
        let mut fake = FakeStream::default();
        let mut capped = PauseCap::new(&mut fake, 0);
        for chunk in clip.chunks(1_234) {
            capped.feed(chunk).unwrap();
        }
        capped.finalize().unwrap();
        assert_eq!(capped.dropped, 0);
        assert_eq!(fake.fed, clip);
    }

    #[test]
    fn a_dictation_stream_hears_trailing_silence_before_it_finalizes() {
        // #657: without trailing silence the stream dropped the closing
        // full stop or question mark the whole clip kept.
        let clip = tone(16_000);
        let mut fake = FakeStream::default();
        let mut capped = PauseCap::new(&mut fake, 6_400);
        for chunk in clip.chunks(1_234) {
            capped.feed(chunk).unwrap();
        }
        assert_eq!(
            capped.finalize(),
            Ok(format!("streamed {}", 16_000 + 6_400))
        );
        assert_eq!(capped.dropped, 0);
        assert_eq!(fake.fed[..16_000], clip[..]);
        assert!(fake.fed[16_000..].iter().all(|&sample| sample == 0.0));
    }

    fn heard(samples: usize) -> Partial {
        Partial {
            stable: format!("heard {samples}"),
            tentative: String::new(),
        }
    }

    /// Where the worker publishes the text of the stream `commands` began.
    fn begun_partials(commands: &Receiver<Command>) -> Option<Partials> {
        match commands.try_recv() {
            Ok(Command::Begin { partials, .. }) => partials,
            _ => panic!("a stream began"),
        }
    }

    /// Drives stream 1 through the commands queued so far, then a cancel.
    fn drive_queued(
        commands: &Receiver<Command>,
        stream: &mut FakeStream,
        partials: Option<&Partials>,
    ) -> FeedStats {
        let (relay, relayed) = mpsc::channel();
        for command in commands.try_iter() {
            relay.send(command).unwrap();
        }
        relay.send(Command::Cancel { stream: 1 }).unwrap();
        let mut stats = FeedStats::default();
        let ended = drive_stream(
            1,
            stream,
            &relayed,
            &AtomicBool::new(false),
            partials,
            &mut VecDeque::new(),
            &mut stats,
        );
        assert!(matches!(ended, StreamEnd::Cancelled));
        stats
    }

    #[test]
    fn a_dictation_stream_publishes_its_text_after_each_feed() {
        let (engine, commands) = Engine::for_test();
        let mut stream = engine.begin().unwrap();
        let partials = begun_partials(&commands).expect("live text gives the stream a slot");
        assert_eq!(
            stream.take_partial(),
            None,
            "nothing is published before a feed"
        );
        stream.feed(&[0.1; 4]);
        stream.feed(&[0.2; 2]);

        let stats = drive_queued(&commands, &mut FakeStream::default(), Some(&partials));
        assert_eq!(stats.partials, 2);
        // The owner reads the newest text once, then nothing until it changes.
        assert_eq!(stream.take_partial(), Some(heard(6)));
        assert_eq!(stream.take_partial(), None);
        assert!(!partials.publish(heard(6)), "the same text is not a change");
        assert_eq!(stream.take_partial(), None);
        assert!(partials.publish(heard(7)));
        assert_eq!(stream.take_partial(), Some(heard(7)));
    }

    #[test]
    fn without_live_text_a_dictation_stream_publishes_nothing() {
        let (engine, commands) = Engine::for_test();
        let engine = engine.without_live_text();
        let mut stream = engine.begin().unwrap();
        assert!(begun_partials(&commands).is_none());
        stream.feed(&[0.1; 4]);
        let stats = drive_queued(&commands, &mut FakeStream::default(), None);
        assert_eq!(stats.partials, 0);
        assert_eq!(stream.take_partial(), None);
    }

    #[test]
    fn each_dictation_stream_has_its_own_text() {
        let (engine, commands) = Engine::for_test();
        let mut first = engine.begin().unwrap();
        let first_partials = begun_partials(&commands).unwrap();
        // An older capture's stream is still finishing when a new one opens.
        let mut second = engine.begin().unwrap();
        let second_partials = begun_partials(&commands).unwrap();
        assert!(first_partials.publish(heard(4)));
        assert_eq!(
            second.take_partial(),
            None,
            "the new capture shows no old text"
        );
        assert_eq!(first.take_partial(), Some(heard(4)));
        assert!(second_partials.publish(heard(1)));
        assert_eq!(second.take_partial(), Some(heard(1)));
        assert_eq!(first.take_partial(), None);
    }

    #[test]
    fn a_failed_feed_clears_the_text_and_stops_publishing() {
        let (engine, commands) = Engine::for_test();
        let mut stream = engine.begin().unwrap();
        let partials = begun_partials(&commands).unwrap();
        stream.feed(&[0.1; 4]);
        stream.feed(&[0.2; 4]);
        stream.feed(&[0.3; 4]);
        let mut failing = FakeStream {
            fail_after: Some(4),
            ..FakeStream::default()
        };
        let stats = drive_queued(&commands, &mut failing, Some(&partials));
        assert_eq!(stats.partials, 1, "the first feed's text, then nothing");
        assert_eq!(failing.fed.len(), 4, "a failed stream is fed no more");
        // The plate hides instead of freezing on the last words.
        assert_eq!(stream.take_partial(), Some(Partial::default()));
    }

    fn begun_yielding(commands: &Receiver<Command>) -> Yielding {
        match commands.try_recv() {
            Ok(Command::BeginYielding { yielding, .. }) => yielding,
            _ => panic!("a yielding stream began"),
        }
    }

    #[test]
    fn a_yielding_stream_reports_partials_and_finishes_through_the_stream() {
        let (engine, commands) = Engine::for_test();
        let mut stream = engine.begin_yielding().unwrap();
        let yielding = begun_yielding(&commands);
        stream.feed(&[0.1; 4]);
        assert_eq!(stream.in_flight(), 4);
        let finishing = thread::spawn(move || stream.finish(Duration::from_secs(5)));
        // The finish is sent from the other thread; wait for it to queue.
        let mut queued = Vec::new();
        while queued.len() < 2 {
            queued.push(commands.recv().unwrap());
        }
        let (relay, relayed) = mpsc::channel();
        for command in queued {
            relay.send(command).unwrap();
        }
        let mut fake = FakeStream::default();
        let interrupted = drive_yielding(1, &mut fake, &relayed, &yielding, &mut VecDeque::new());
        assert!(interrupted.is_none());
        assert_eq!(finishing.join().unwrap(), Ok("streamed 4".into()));
        assert_eq!(yielding.consumed.load(Ordering::Acquire), 4);
        assert_eq!(
            yielding.partials.latest(),
            Some(Partial {
                stable: "heard 4".into(),
                tentative: String::new(),
            })
        );
    }

    #[test]
    fn a_dictation_request_closes_a_yielding_stream_and_runs_next() {
        let (engine, commands) = Engine::for_test();
        let mut stream = engine.begin_yielding().unwrap();
        let yielding = begun_yielding(&commands);
        stream.feed(&[0.1; 4]);
        let generation = engine.requests.load(Ordering::Acquire);
        let _chunk = engine
            .foreground(vec![0.0; 3], Box::new(|_, _| Ok(text_transcript("chunk"))))
            .unwrap();
        assert_eq!(
            engine.requests.load(Ordering::Acquire),
            generation + 1,
            "foreground work is a dictation-class request"
        );
        let _dictation = engine.begin().unwrap();
        stream.feed(&[0.2; 4]);

        let mut fake = FakeStream::default();
        let mut deferred = VecDeque::new();
        let interrupted = drive_yielding(1, &mut fake, &commands, &yielding, &mut deferred);
        assert!(matches!(
            interrupted,
            Some(Command::Begin { stream: 2, .. })
        ));
        assert!(!stream.open(), "dictation closed the stream");
        assert_eq!(fake.fed.len(), 4);
        assert!(matches!(
            deferred.as_slices().0,
            [Command::Foreground { clip, .. }] if clip.len() == 3
        ));
        // A closed stream answers at once; its owner transcribes offline.
        assert_eq!(stream.finish(Duration::from_secs(5)), Err(PREEMPTED.into()));
    }

    #[test]
    fn an_open_dictation_stream_refuses_a_yielding_stream_instead_of_ending() {
        let (engine, commands) = Engine::for_test();
        let mut dictation = engine.begin().unwrap();
        assert!(matches!(
            commands.try_recv().unwrap(),
            Command::Begin { stream: 1, .. }
        ));
        let yielding = engine.begin_yielding().unwrap();
        dictation.feed(&[0.1; 4]);
        let _pending = engine
            .finish_with(
                Some(dictation),
                vec![0.0; 4],
                Box::new(|_, _| panic!("the dictation must not be retranscribed offline")),
            )
            .unwrap();
        let mut fake = FakeStream::default();
        let ended = drive_stream(
            1,
            &mut fake,
            &commands,
            &AtomicBool::new(false),
            None,
            &mut VecDeque::new(),
            &mut FeedStats::default(),
        );
        let StreamEnd::Finished { result, .. } = ended else {
            panic!("the dictation stream did not finish");
        };
        assert_eq!(result, Ok("streamed 4".to_string()));
        assert!(!yielding.open());
    }

    const SHORT: Deadlines = Deadlines {
        stream_finish: Duration::from_millis(50),
        job: Duration::from_millis(100),
        job_seconds_per_audio_second: 0,
        shutdown: Duration::from_millis(100),
    };

    fn silent() -> Offline {
        Box::new(|_, _| Ok(Transcript::default()))
    }

    /// A transcript of just text, as a test's offline run returns.
    fn text_transcript(text: &str) -> Transcript {
        Transcript {
            text: text.into(),
            ..Transcript::default()
        }
    }

    /// Takes the next finish the test engine would have run.
    fn next_job(commands: &Receiver<Command>) -> Job {
        commands
            .try_iter()
            .find_map(|command| match command {
                Command::Finish { job, .. } => Some(job),
                _ => None,
            })
            .expect("a finish was sent")
    }

    #[test]
    fn a_started_job_that_never_ends_fails_instead_of_hanging() {
        let (engine, commands) = Engine::for_test();
        let engine = engine.with_deadlines(SHORT);
        for streamed in [false, true] {
            let stream = streamed.then(|| engine.begin().expect("the test engine streams"));
            let pending = engine
                .finish_with(stream, vec![0.0; 1_600], silent())
                .unwrap();
            let job = next_job(&commands);
            job.start();
            let started = Instant::now();
            assert_eq!(pending.wait().unwrap_err(), "transcription timed out");
            assert!(started.elapsed() < Duration::from_secs(2));
            assert!(job.cancel.is_cancelled(), "a timeout aborts the stuck run");
            engine.stalled.store(false, Ordering::Release);
        }
    }

    #[test]
    fn the_job_clock_does_not_count_time_queued_behind_other_work() {
        let (engine, commands) = Engine::for_test();
        let engine = engine.with_deadlines(SHORT);
        let pending = engine.finish_with(None, Vec::new(), silent()).unwrap();
        let job = next_job(&commands);
        let worker = thread::spawn(move || {
            // Three job deadlines pass before the engine reaches the job.
            thread::sleep(SHORT.job * 3);
            job.start();
            job.finish(Ok(Transcribed {
                transcript: text_transcript("late but fine"),
                source: Source::Offline,
                finish_ms: 0,
            }));
        });
        assert_eq!(pending.wait().unwrap().text(), "late but fine");
        worker.join().unwrap();
    }

    #[test]
    fn a_timed_out_job_cancels_only_itself_not_the_next_capture() {
        let (engine, commands) = Engine::for_test();
        let engine = engine.with_deadlines(SHORT);
        let first = engine.finish_with(None, Vec::new(), silent()).unwrap();
        let first_job = next_job(&commands);
        first_job.start();
        // Capture 2 begins while job 1 is still running.
        let second = engine.begin().expect("the test engine streams");
        assert_eq!(first.wait().unwrap_err(), "transcription timed out");
        assert!(first_job.cancel.is_cancelled());
        assert!(!second.cancel.is_cancelled(), "capture 2's stream survives");
        let Some(Command::Begin { cancel, .. }) = commands.try_iter().next() else {
            panic!("capture 2 began a stream");
        };
        assert!(!cancel.is_cancelled());
        // A capture that begins after the timeout is untouched too.
        let third = engine.begin().expect("the test engine streams");
        assert!(!third.cancel.is_cancelled());
    }

    #[test]
    fn jobs_queued_behind_a_wedged_run_fail_instead_of_waiting_forever() {
        let (engine, commands) = Engine::for_test();
        let engine = engine.with_deadlines(SHORT);
        let wedged = engine.finish_with(None, Vec::new(), silent()).unwrap();
        let queued = engine.finish_with(None, Vec::new(), silent()).unwrap();
        let wedged_job = next_job(&commands);
        let queued_job = next_job(&commands);
        wedged_job.start();
        assert_eq!(wedged.wait().unwrap_err(), "transcription timed out");
        let started = Instant::now();
        assert_eq!(queued.wait().unwrap_err(), "transcription engine is stuck");
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(
            queued_job.cancel.is_cancelled(),
            "the engine skips it later"
        );
    }

    #[test]
    fn the_job_deadline_grows_with_the_clip() {
        let (engine, _commands) = Engine::for_test_without_streaming();
        let pending = engine
            .finish_with(None, vec![0.0; 16_000 * 30], silent())
            .unwrap();
        assert_eq!(
            pending.job,
            JOB_DEADLINE + Duration::from_secs(30 * u64::from(JOB_SECONDS_PER_AUDIO_SECOND))
        );
    }

    #[test]
    fn a_workerless_engine_refuses_jobs_instead_of_hanging() {
        // A test engine has no worker: waiting on it would poll forever, so
        // jobs and background work fail at once.
        let (engine, _commands) = Engine::for_test();
        assert!(engine.transcribe_clip(Vec::new()).is_err());
        assert!(engine.background(|_| ()).is_err());
        assert!(!engine.diarizes());
        assert!(engine.diarize(&[]).is_err());
    }

    #[test]
    fn a_diarize_job_sends_one_command_and_answers_through_its_reply() {
        let (engine, commands) = Engine::for_test();
        let engine = engine.with_test_diarize();
        let pending = engine.diarize_job(&[0.0; 1_600]).unwrap();
        let (pcm, job) = commands
            .try_iter()
            .find_map(|command| match command {
                Command::Diarize { pcm, job } => Some((pcm, job)),
                _ => None,
            })
            .expect("a diarize job was sent");
        assert_eq!(pcm.len(), 1_600);
        job.start();
        job.finish_turns(Ok(vec![super::SpeakerTurn {
            start_ms: 0,
            end_ms: 100,
            speaker: 1,
            p: 0.9,
        }]));
        let turns = pending.wait().unwrap();
        assert_eq!(
            turns.as_slice(),
            [{
                super::SpeakerTurn {
                    start_ms: 0,
                    end_ms: 100,
                    speaker: 1,
                    p: 0.9,
                }
            }]
        );
    }

    #[test]
    fn a_queued_diarize_job_behind_a_wedged_engine_fails_fast() {
        let (reply, receiver) = mpsc::channel();
        // Hold the sender open: the waiter must learn the engine is wedged
        // from the stalled flag, not from a dropped channel.
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(5));
            drop(reply);
        });
        let pending = PendingTurns {
            reply: receiver,
            cancel: CancelToken::new(),
            stalled: Arc::new(AtomicBool::new(true)),
            job: SHORT.job,
        };
        assert_eq!(pending.wait().unwrap_err(), "transcription engine is stuck");
    }
}
