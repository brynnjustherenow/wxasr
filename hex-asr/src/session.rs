//! A push-based streaming recognition session over a shared [`Engine`],
//! shaped to map 1:1 onto a unified async layer's session contract
//! (wxasr-async's `AsrSession`: `push(&mut self, pcm: &[i16], out: &mut dyn
//! AsrEvents)` / `finish(&mut self, out)`).
//!
//! This crate adapts to that contract; nothing upstream changes:
//!
//! - Audio in: 16 kHz mono **i16**, any chunk size ([`StreamingSession::push`]).
//! - Events out: only through a sink ([`SessionEvents`]), as
//!   [`SessionEvent::Partial`] while recording and exactly one
//!   [`SessionEvent::Final`] at [`StreamingSession::finish`]. An empty final
//!   text emits nothing. Hex semantics: **no context rotation** — one
//!   session, one final; multi-`Final` rotation is another backend's own
//!   behavior.
//! - [`SessionEvents::emit`] returning `false` means the consumer is gone:
//!   the session cancels its stream and stops computing (later pushes and a
//!   `finish` do nothing).
//!
//! A wxasr-async backend is then a thin adapter on *its* side:
//!
//! ```text
//! struct HexBackend { engine: Arc<hex_asr::Engine> }   // AsrBackend
//! fn open(&self, cfg) -> Result<Box<dyn AsrSession>> {
//!     Ok(Box::new(hex_asr::StreamingSession::open(
//!         self.engine.clone(), cfg.partial,
//!     )))  // + a 4-line AsrEvents → SessionEvents adapter
//! }
//! ```
//!
//! `Config::partial_min_fresh` is wxasr-only (hex streams throttle
//! themselves); `Config::rotate` the hex backend implements *on top of* this
//! session: the native stream is one-shot (no split-off or incremental
//! finalize API), so a rotation is a `finish` (which drains the decode lag
//! into this segment) followed by a fresh `open` — this session itself never
//! rotates.

use std::sync::Arc;

use crate::engine::{Engine, LiveStream};

/// One recognition event, as a session reports it through a
/// [`SessionEvents`] sink. Mirrors the unified layer's event payloads.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionEvent {
    /// The current segment's provisional text (the stream's committed
    /// prefix plus its volatile hypothesis). Only while recording; never
    /// pasted or kept by the caller.
    Partial(String),
    /// The one finalized result of this session's complete clip: text,
    /// the model-detected language when it predicted one, and any
    /// segment/word timings it produced. A stream finish carries text only.
    Final(crate::Transcript),
}

/// A session's event exit, provided by the driver. `emit` returns `false`
/// when the consumer has closed; the session then stops computing and need
/// not emit again (further events would be dropped anyway).
pub trait SessionEvents {
    fn emit(&mut self, event: SessionEvent) -> bool;
}

/// One streaming recognition session over a shared engine: feed 16 kHz mono
/// i16, take partials through a sink, finish once with the complete clip.
///
/// The session keeps every pushed sample as the authoritative clip; the
/// finish carries all of it, so a failed, stalled, or empty stream falls
/// back to an offline transcription of the whole clip (Hex semantics). A
/// model that cannot stream on its device opens in offline mode: pushes
/// only accumulate, and the finish transcribes the clip once.
pub struct StreamingSession {
    engine: Arc<Engine>,
    stream: Option<LiveStream>,
    /// The authoritative clip: every pushed sample, in f32.
    clip: Vec<f32>,
    /// i16 → f32 conversion scratch.
    converted: Vec<f32>,
    /// The latest partial's committed prefix, which only grows within one
    /// stream. A rotation policy reads it through
    /// [`committed_text`](Self::committed_text) to spot new sentence-end
    /// punctuation.
    committed: String,
    /// Whether pushes report partials at all.
    partials: bool,
    /// The consumer closed; stop computing.
    dead: bool,
}

impl StreamingSession {
    /// Opens a session on `engine`, reporting partial text when `partials`
    /// is on. `None`-streaming models open in offline mode (see the type
    /// docs); the engine thread itself must already be running.
    pub fn open(engine: Arc<Engine>, partials: bool) -> Self {
        let stream = engine.begin();
        if partials && stream.is_none() {
            tracing::info!(
                "this model cannot stream on its device; the session only accumulates, so no partial text"
            );
        }
        Self {
            engine,
            stream,
            clip: Vec::new(),
            converted: Vec::new(),
            committed: String::new(),
            partials,
            dead: false,
        }
    }

    /// Feeds 16 kHz mono i16 of any chunk size, reporting the stream's
    /// latest partial text when it changed.
    pub fn push(&mut self, pcm: &[i16], out: &mut dyn SessionEvents) {
        if self.dead || pcm.is_empty() {
            return;
        }
        // `i16_to_f32` appends: clear the scratch so it holds only this
        // chunk, or every push re-feeds everything before it.
        self.converted.clear();
        crate::audio::i16_to_f32(pcm, &mut self.converted);
        self.clip.extend_from_slice(&self.converted);
        let Some(stream) = &mut self.stream else {
            return;
        };
        stream.feed(&self.converted);
        if self.partials
            && let Some(partial) = stream.take_partial()
        {
            let text = partial_text(&partial);
            self.committed = partial.stable;
            if !text.is_empty() && !out.emit(SessionEvent::Partial(text)) {
                self.close();
            }
        }
    }

    /// Finalizes with the complete clip and emits the session's one
    /// [`SessionEvent::Final`]; an empty final text emits nothing. After a
    /// closed consumer (an `emit` that returned `false`), or a second call,
    /// it does nothing. This blocks for the transcription, bounded by the
    /// engine's stream-finish and job deadlines.
    pub fn finish(&mut self, out: &mut dyn SessionEvents) {
        if self.dead {
            return;
        }
        self.dead = true;
        let stream = self.stream.take();
        let clip = std::mem::take(&mut self.clip);
        let pending = match self.engine.finish(stream, clip) {
            Ok(pending) => pending,
            Err(error) => {
                tracing::warn!(%error, "could not submit the final transcription");
                return;
            }
        };
        match pending.wait() {
            Ok(transcribed) => {
                let transcript = transcribed.transcript;
                if !transcript.text.trim().is_empty() {
                    out.emit(SessionEvent::Final(transcript));
                }
            }
            Err(reason) => {
                tracing::warn!(%reason, "the final transcription failed");
            }
        }
    }

    /// Cancels the stream and stops computing, as a closed consumer asks.
    /// Dropping the [`LiveStream`] cancels it on the engine.
    fn close(&mut self) {
        self.dead = true;
        self.stream = None;
        self.clip.clear();
    }

    /// Samples pushed so far, as the finish will carry them.
    pub fn pushed_samples(&self) -> usize {
        self.clip.len()
    }

    /// The stream's committed text so far: the stable prefix of its latest
    /// partial, which only grows within one stream. Empty for an offline
    /// session (nothing streams) and until the first partial arrives. A
    /// rotation policy scans its growth for sentence-end punctuation.
    pub fn committed_text(&self) -> &str {
        &self.committed
    }

    /// Whether this session streams on its device (an offline session only
    /// accumulates, and finalizes from the whole clip).
    pub fn is_streaming(&self) -> bool {
        self.stream.is_some()
    }
}

/// A partial's committed prefix and volatile hypothesis as one text.
fn partial_text(partial: &crate::Partial) -> String {
    if partial.stable.is_empty() {
        partial.tentative.trim().to_string()
    } else if partial.tentative.is_empty() {
        partial.stable.trim().to_string()
    } else {
        format!("{} {}", partial.stable.trim(), partial.tentative.trim())
            .trim()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Command;
    use std::sync::mpsc::Receiver;

    /// A sink that records events and answers `alive` from now on.
    struct Recorder {
        events: Vec<(bool, SessionEvent)>,
        alive: bool,
    }

    impl SessionEvents for Recorder {
        fn emit(&mut self, event: SessionEvent) -> bool {
            self.events.push((self.alive, event));
            self.alive
        }
    }

    fn test_session() -> (Arc<Engine>, StreamingSession, Receiver<Command>) {
        let (engine, commands) = Engine::for_test();
        let engine = Arc::new(engine);
        let session = StreamingSession::open(engine.clone(), true);
        assert!(session.is_streaming(), "the test engine streams");
        (engine, session, commands)
    }

    #[test]
    fn a_session_feeds_i16_as_f32_in_order() {
        let (_engine, mut session, commands) = test_session();
        session.push(
            &[0, 16, -256],
            &mut Recorder {
                events: Vec::new(),
                alive: true,
            },
        );
        session.push(
            &[1024],
            &mut Recorder {
                events: Vec::new(),
                alive: true,
            },
        );
        assert_eq!(session.pushed_samples(), 4);
        let received = commands.try_iter().collect::<Vec<_>>();
        assert!(
            matches!(
                received.as_slice(),
                [
                    Command::Begin { stream: 1, .. },
                    Command::Feed { stream: 1, samples },
                    Command::Feed { stream: 1, .. },
                ] if samples.as_slice() == [0.0, 16.0 / 32768.0, -256.0 / 32768.0]
            ),
            "the session began a stream and fed converted f32 in order"
        );
    }

    #[test]
    fn an_offline_model_accumulates_without_a_stream() {
        let (engine, commands) = Engine::for_test_without_streaming();
        let mut session = StreamingSession::open(Arc::new(engine), true);
        assert!(!session.is_streaming());
        session.push(
            &[0; 1600],
            &mut Recorder {
                events: Vec::new(),
                alive: true,
            },
        );
        assert_eq!(session.pushed_samples(), 1600);
        assert!(commands.try_iter().next().is_none(), "nothing is streamed");
    }

    #[test]
    fn a_closed_consumer_stops_the_session() {
        let (_engine, mut session, commands) = test_session();
        let mut dead = Recorder {
            events: Vec::new(),
            alive: false,
        };
        // The worker would publish a partial; simulate one by publishing
        // through the stream the session holds.
        let partials = match commands.try_recv() {
            Ok(Command::Begin { partials, .. }) => partials,
            _ => panic!("a stream began"),
        };
        let published = crate::Partial {
            stable: "hello".into(),
            tentative: String::new(),
        };
        partials
            .as_ref()
            .expect("partials are on")
            .publish(published.clone());
        session.push(&[0; 100], &mut dead);
        // The partial was emitted once, its false closed the session…
        assert_eq!(
            dead.events,
            vec![(false, SessionEvent::Partial("hello".into()))]
        );
        // …its stream is cancelled (dropped) at once, and nothing more is
        // fed: the next push is a no-op.
        session.push(
            &[0; 100],
            &mut Recorder {
                events: Vec::new(),
                alive: true,
            },
        );
        let received = commands.try_iter().collect::<Vec<_>>();
        assert!(
            matches!(
                received.as_slice(),
                [
                    Command::Feed { stream: 1, .. },
                    Command::Cancel { stream: 1 }
                ]
            ),
            "the feed went out, then the closed session cancelled"
        );
    }

    #[test]
    fn a_finish_after_a_closed_consumer_does_nothing() {
        let (_engine, mut session, commands) = test_session();
        // A session learns the consumer closed only from an emit that
        // returns false, so publish a partial first (as the worker would).
        let partials = match commands.try_recv() {
            Ok(Command::Begin { partials, .. }) => partials,
            _ => panic!("a stream began"),
        };
        partials
            .as_ref()
            .expect("partials are on")
            .publish(crate::Partial {
                stable: "hi".into(),
                tentative: String::new(),
            });
        session.push(
            &[0; 100],
            &mut Recorder {
                events: Vec::new(),
                alive: false,
            },
        );
        let mut sink = Recorder {
            events: Vec::new(),
            alive: true,
        };
        session.finish(&mut sink);
        assert!(sink.events.is_empty(), "no final for a dead session");
        // The dead session submitted no job either: only its feed and the
        // stream's cancellation went out.
        let received = commands.try_iter().collect::<Vec<_>>();
        assert!(matches!(
            received.as_slice(),
            [
                Command::Feed { stream: 1, .. },
                Command::Cancel { stream: 1 }
            ]
        ));
    }

    #[test]
    fn partial_text_joins_stable_and_tentative() {
        assert_eq!(
            partial_text(&crate::Partial {
                stable: "hello".into(),
                tentative: " wor".into(),
            }),
            "hello wor"
        );
        assert_eq!(
            partial_text(&crate::Partial {
                stable: String::new(),
                tentative: " hi ".into(),
            }),
            "hi"
        );
        assert_eq!(
            partial_text(&crate::Partial {
                stable: " hi ".into(),
                tentative: String::new(),
            }),
            "hi"
        );
        assert_eq!(
            partial_text(&crate::Partial::default()),
            "",
            "empty partials never emit"
        );
    }

    /// The engine and a held session must cross threads: the engine because
    /// a unified-layer backend (`Send + Sync`) holds `Arc<Engine>`, the
    /// session because its worker drives it on a dedicated thread.
    #[test]
    fn engine_and_session_are_send_and_sync() {
        fn assert_both<T: Send + Sync>() {}
        assert_both::<Engine>();
        assert_both::<StreamingSession>();
        fn assert_send<T: Send>() {}
        assert_send::<crate::LiveStream>();
    }
}
