//! wxasr-async —— 流式 ASR 的统一异步层:专用识别线程 + 通道桥。
//!
//! 统一层只做桥:音频经 std mpsc 进入([`AsrHandle::push`] 同步、非阻塞,
//! 可直接在采集回调里调用),事件经 tokio mpsc 流出到 async 侧,async
//! 只做分发;识别线程驱动后端([`AsrBackend`])创建的会话
//! ([`AsrSession`]),会话语义(中间结果节流、上下文轮换、定稿兜底)
//! 全部由后端自持,不进 async 执行器。
//!
//! wxasr 是第一个后端([`backend_wxasr::WxasrBackend`]):VAD、partial
//! 节流与 [`policy`] 轮换都住在那里,经 [`start`] 开箱即用;hex-asr
//! 后端([`backend_hex::HexBackend`],`hex` feature)实现同一契约,自带
//! 轮换(native 流一次性,轮换 = 老流 finish + 新流)与 Silero VAD
//! endpointing。各后端经 [`start_with`] 接入,共享同一套线程/通道桥与
//! 事件流(会话语义见各后端模块文档)。
//!
//! # 事件流
//!
//! 事件序:`(Partial* Commit)* Partial* End?` → 通道关闭。
//!
//! - [`AsrEvent::Partial`][]:当前段的中间结果(tentative 可回改,
//!   显示用)。wxasr 后端已按计算成本自动节流(新增 ≥
//!   `partial_min_fresh` 帧,且距上次 ≥ 3× 上次计算耗时,慢机自动降频)
//! - [`AsrEvent::Commit`][]:一段已定稿——稳定前缀提交 + 切断,由上下文
//!   轮换产生,会话继续(空文本不发)。之后的 `Partial` 从空开始
//! - [`AsrEvent::End`][]:会话结束([`AsrHandle::finish`] 或所有句柄
//!   drop)补定的最后一段(空文本不发);最多一个。会话结束仍以
//!   **通道关闭**(`recv` 返回 `None`)为准,`End` 不是关闭事件
//!
//! 事件载荷是 [`AsrText`]:除 `content` 外的字段按模型能力填充,
//! 不支持为 `None`/空(wxasr 后端只填 `content`)。
//!
//! # 音频格式
//!
//! 音频格式由调用方在 [`start_with`] 声明([`AudioIn`]),之后所有
//! [`AsrHandle::push`] 的 PCM 都按此解释——统一层不写死采样率,
//! 降混与重采样在各后端内部完成,调用方无感。16 kHz 单声道
//! ([`AudioIn::MONO_16K`])零转换直通。
//!
//! 完整转写 = 按顺序拼接所有 `Commit` 与 `End`。
//!
//! # 入门
//!
//! ```no_run
//! # async fn demo(model: std::sync::Arc<wxasr::Model>, mic_pcm: &[i16]) {
//! use wxasr_async::{start, AsrEvent};
//!
//! let (handle, mut rx) = start(model, Default::default());
//! handle.push(mic_pcm);           // 采集回调里喂声明的格式(默认 16kHz 单声道 i16)
//! handle.finish();                // 结束识别
//!
//! while let Some(ev) = rx.recv().await {
//!     match ev {
//!         AsrEvent::Partial(t) => { /* 当前段边说边显 */ }
//!         AsrEvent::Commit(t) => println!("{}", t.content), // 一段定稿,会话继续
//!         AsrEvent::End(t) => println!("{}", t.content),    // 最后一段,随后通道关闭
//!     }
//! }
//! // 通道关闭:会话结束
//! # }
//! ```
//!
//! 换其他后端只需把 [`start`] 换成 [`start_with`],事件流不变。

use std::sync::Arc;
use tokio::sync::mpsc;
use wxasr::Model;
#[cfg(feature = "hex")]
pub mod backend_hex;
pub mod backend_wxasr;
pub mod policy;
pub mod preprocess;
use policy::RotatePolicy;

/// 词级时间戳:毫秒相对会话起点,`end_ms ≥ start_ms`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrWord {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// 一条分段:时间戳加可选的说话人归属
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// 1-based 说话人编号;模型未归属(或无 diarization)→ `None`
    pub speaker: Option<u32>,
}

/// 一次识别结果,[`AsrEvent::Partial`] 与 [`AsrEvent::Commit`]/
/// [`AsrEvent::End`] 共用。
///
/// 除 `content` 外的字段按模型能力填充:模型(或后端)不支持时为
/// `None`/空——wxasr 后端只填 `content`。`Partial` 通常也只有
/// `content`(中间结果是文本增量);语言/时间戳在 `Commit`/`End` 到齐
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AsrText {
    /// 识别文本,唯一必填字段
    pub content: String,
    /// 模型自报的语言(ISO code);无语言识别 → `None`
    pub language: Option<String>,
    /// 整段归属的说话人(1-based);无归属或混合 → `None`
    pub speaker: Option<u32>,
    /// 分段(时间戳 + 可选说话人);模型不给 → 空
    pub segments: Vec<AsrSegment>,
    /// 词级时间戳;模型不给 → 空
    pub words: Vec<AsrWord>,
}

impl AsrText {
    /// 只有文本的结果,不支持富字段的后端用
    pub fn plain(content: impl Into<String>) -> Self {
        AsrText {
            content: content.into(),
            ..AsrText::default()
        }
    }
}

/// 识别事件(按时间顺序产生):`(Partial* Commit)* Partial* End?` →
/// 通道关闭
#[derive(Debug, Clone)]
pub enum AsrEvent {
    /// 当前段的中间结果(已节流,仅当开启 `Config::partial`);tentative
    /// 可回改,只做显示。每次 [`Commit`](Self::Commit)/[`End`](Self::End)
    /// 之后从空开始
    Partial(AsrText),
    /// 一段已定稿:稳定前缀提交 + 切断,由上下文轮换产生,**会话继续**
    /// (空文本不发)
    Commit(AsrText),
    /// 会话结束(`finish`/句柄全 drop)补定的最后一段(空文本不发,
    /// 最多一个)。不是关闭事件——会话结束以通道关闭为准
    End(AsrText),
}

/// 调用方音频的格式:会话创建时声明一次,之后所有
/// [`AsrHandle::push`] 的 PCM 都按此解释(交错 i16)。
///
/// 统一层不写死采样率:各后端在内部降混/重采样成自己需要的格式,
/// 调用方无感;16 kHz 单声道零转换直通
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioIn {
    pub sample_rate: u32,
    /// 声道数(交错存储)
    pub channels: u16,
}

impl AudioIn {
    /// 16 kHz 单声道,历史默认(wxasr 原生格式,直通)
    pub const MONO_16K: AudioIn = AudioIn {
        sample_rate: 16_000,
        channels: 1,
    };
}

/// 会话的事件出口,由统一 worker 提供给后端。
///
/// `emit` 返回 `false` 表示消费端已关闭,后端应尽快收尾:停止后续
/// 计算,不必再发事件(发了也会被丢弃)
pub trait AsrEvents {
    /// 发送一个事件;返回 `false` 表示消费端已关闭
    fn emit(&mut self, ev: AsrEvent) -> bool;
}

/// 一个流式识别会话,由统一 worker 在专用识别线程上驱动。实现方
/// 持有全部会话语义(中间结果节流、轮换/定稿、兜底),不进 async
/// 执行器
pub trait AsrSession: Send {
    /// 喂 [`AudioIn`] 所声明格式的 i16 PCM(交错,任意块大小);
    /// 后端已在内部转换成自己需要的格式
    fn push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents);
    /// 定稿最后一段并结束(空文本不发事件)
    fn finish(&mut self, out: &mut dyn AsrEvents);
}

/// ASR 后端:为每次启动创建一个 [`AsrSession`]
pub trait AsrBackend: Send + Sync {
    /// 在识别线程内调用。`audio` 是调用方声明的音频格式,会话的
    /// [`AsrSession::push`] 按它解释,后端自行降混/重采样。失败时
    /// 会话立即结束:事件通道关闭、不产生任何事件(消费端看到空会话)
    fn open(&self, cfg: &Config, audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>>;
}

/// 运行配置。
///
/// 字段分两层:`rotate`/`partial`/`partial_min_fresh` 是会话语义,由
/// wxasr 后端解释(其他后端按自身语义处理或忽略);`event_capacity`/
/// `thread_name` 属统一层,与后端无关。后端自身的专属配置放后端的
/// 构造参数,不进 `Config`
#[derive(Debug, Clone)]
pub struct Config {
    /// 上下文轮换策略,决定何时定稿当前流并重建以约束资源增长
    /// (wxasr 后端解释)
    pub rotate: RotatePolicy,
    /// 是否产生 [`AsrEvent::Partial`] (wxasr 后端解释)
    pub partial: bool,
    /// 中间结果的最小新增帧数(10ms/帧;wxasr 后端解释)
    pub partial_min_fresh: usize,
    /// 事件通道容量(统一层:消费方卡死时识别线程最多积压这么多事件)
    pub event_capacity: usize,
    /// 识别线程名(统一层)。`None` 时自动生成 `wxasr-<uuidv7>`(时间
    /// 有序,天然唯一;crate 历史默认,统一层沿用),多流并发无需配置
    /// 也不会重名;需要可读名字(如 "wxasr-desktop")时显式传入。
    pub thread_name: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            rotate: RotatePolicy::default(),
            partial: true,
            partial_min_fresh: 30,
            event_capacity: 64,
            thread_name: None,
        }
    }
}

enum AudioMsg {
    Pcm(Vec<i16>),
    End,
}

/// 音频喂入端。`Clone` 后可多处持有(如采集回调与停止按钮);
/// 任一处调用 [`finish`](Self::finish) 即结束识别,之后的 `push` 静默忽略。
#[derive(Clone)]
pub struct AsrHandle {
    tx: std::sync::mpsc::Sender<AudioMsg>,
}

impl AsrHandle {
    /// 喂入一段 i16 PCM(任意块大小),按 [`start_with`] 时声明的
    /// [`AudioIn`] 解释。同步、非阻塞、不丢数据,可直接在采集回调里调用。
    pub fn push(&self, pcm: &[i16]) {
        let _ = self.tx.send(AudioMsg::Pcm(pcm.to_vec()));
    }

    /// 结束识别:识别线程定稿最后一段并发送 [`AsrEvent::End`],
    /// 随后关闭事件通道。消耗 self;所有句柄 drop 时同样触发结束。
    pub fn finish(self) {
        let _ = self.tx.send(AudioMsg::End);
    }
}

/// 用 wxasr 后端启动识别(等价于
/// `start_with(Box::new(WxasrBackend::new(model)), cfg, AudioIn::MONO_16K)`)。
/// 返回(喂入端,事件流)。
///
/// 多个识别流可共享同一 `Arc<Model>`:权重与内部计算线程池全局复用,
/// 每流仅一个 worker 线程与各自的流式状态(特征/VAD)。并发流建议
/// 通过 [`Config::thread_name`] 各自命名。
pub fn start(model: Arc<Model>, cfg: Config) -> (AsrHandle, mpsc::Receiver<AsrEvent>) {
    start_with(
        Box::new(backend_wxasr::WxasrBackend::new(model)),
        cfg,
        AudioIn::MONO_16K,
    )
}

/// 通用入口:用任意 [`AsrBackend`] 启动识别。返回(喂入端,事件流)。
///
/// 统一层 spawn 一个专用识别线程,在其中 `backend.open(&cfg, audio)`
/// 创建会话并驱动:音频经 [`AsrHandle::push`] 进(按 `audio` 声明的
/// 格式解释,后端内部转换),事件经 tokio mpsc 出;
/// [`AsrHandle::finish`] 或所有句柄 drop 时调用 [`AsrSession::finish`]
/// 定稿并关闭事件通道;消费端提前关闭时识别线程停止驱动、直接丢弃
/// 会话(不定稿)。线程默认名沿用 `wxasr-<uuidv7>`,并发流用
/// [`Config::thread_name`] 各自命名。
pub fn start_with(
    backend: Box<dyn AsrBackend>,
    cfg: Config,
    audio: AudioIn,
) -> (AsrHandle, mpsc::Receiver<AsrEvent>) {
    let (audio_tx, audio_rx) = std::sync::mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel(cfg.event_capacity);
    let name = match cfg.thread_name.clone() {
        Some(n) => n,
        // uuidv7:时间有序、并发不重名(Linux 线程名 15 字节截断不影响前缀唯一性)
        None => format!("wxasr-{}", uuid::Uuid::now_v7()),
    };
    std::thread::Builder::new()
        .name(name)
        .spawn(move || worker(backend.as_ref(), cfg, audio, audio_rx, event_tx))
        .expect("识别线程创建失败");
    (AsrHandle { tx: audio_tx }, event_rx)
}

fn worker(
    backend: &dyn AsrBackend,
    cfg: Config,
    audio: AudioIn,
    audio_in: std::sync::mpsc::Receiver<AudioMsg>,
    out: mpsc::Sender<AsrEvent>,
) {
    // open 失败:会话立即结束(通道关闭,不产生事件),见 AsrBackend::open
    let Ok(mut session) = backend.open(&cfg, audio) else {
        return;
    };
    let mut sink = EventSink {
        tx: out,
        closed: false,
    };
    // 消费端提前关闭:直接丢弃会话、不调用 finish,与旧版 worker
    // blocking_send 出错即 return 的行为一致
    let mut consumer_alive = true;
    while let Ok(msg) = audio_in.recv() {
        let AudioMsg::Pcm(pcm) = msg else { break };
        session.push(&pcm, &mut sink);
        if sink.closed {
            consumer_alive = false;
            break;
        }
    }
    if consumer_alive {
        session.finish(&mut sink);
    }
}

/// [`AsrEvents`] 的统一实现:包装 tokio `Sender` 的 `blocking_send`
/// (识别线程不是 async 上下文;有界通道满时阻塞,即事件背压)。
/// 消费端关闭返回 `false` 并记下,worker 据此停止驱动
struct EventSink {
    tx: mpsc::Sender<AsrEvent>,
    closed: bool,
}

impl AsrEvents for EventSink {
    fn emit(&mut self, ev: AsrEvent) -> bool {
        if self.tx.blocking_send(ev).is_err() {
            self.closed = true; // 消费端已关闭
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    /// 假后端:累计音频毫秒,满 100ms 发一个 Partial,finish 发 Final
    struct CountBackend;

    impl AsrBackend for CountBackend {
        fn open(&self, _cfg: &Config, _audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>> {
            Ok(Box::new(CountSession {
                ms: 0,
                partial_sent: false,
            }))
        }
    }

    struct CountSession {
        ms: u32,
        partial_sent: bool,
    }

    impl AsrSession for CountSession {
        fn push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
            self.ms += pcm.len() as u32 / 16;
            if self.ms >= 100 && !self.partial_sent {
                self.partial_sent = true;
                out.emit(AsrEvent::Partial(AsrText::plain("够 100ms 了")));
            }
        }

        fn finish(&mut self, out: &mut dyn AsrEvents) {
            out.emit(AsrEvent::End(AsrText::plain(format!("{}ms", self.ms))));
        }
    }

    #[test]
    fn start_with_drives_backend_until_handle_finish() {
        let (handle, mut rx) =
            start_with(Box::new(CountBackend), Config::default(), AudioIn::MONO_16K);
        handle.push(&[0i16; 1600]); // 100ms
        handle.finish();
        let ev = rx.blocking_recv();
        assert!(
            matches!(ev, Some(AsrEvent::Partial(ref t)) if t.content == "够 100ms 了"),
            "应为 Partial,实际 {ev:?}"
        );
        let ev = rx.blocking_recv();
        assert!(
            matches!(ev, Some(AsrEvent::End(ref t)) if t.content == "100ms"),
            "应为 End,实际 {ev:?}"
        );
        assert!(rx.blocking_recv().is_none()); // 通道关闭 = 会话结束
    }

    #[test]
    fn dropping_all_handles_finishes_session() {
        let (handle, mut rx) =
            start_with(Box::new(CountBackend), Config::default(), AudioIn::MONO_16K);
        handle.push(&[0i16; 320]); // 20ms
        drop(handle);
        let ev = rx.blocking_recv();
        assert!(
            matches!(ev, Some(AsrEvent::End(ref t)) if t.content == "20ms"),
            "应为 End,实际 {ev:?}"
        );
        assert!(rx.blocking_recv().is_none());
    }

    /// 探针会话:记录被调用的事件。push 每块都发事件,以尽快探测
    /// 消费端;`emit` 返回 false 后不再计算(尊重 false)
    struct ProbeSession {
        log: Arc<Mutex<Vec<&'static str>>>,
        dead: bool,
    }

    impl ProbeSession {
        fn record(&self, ev: &'static str) {
            // poison 容忍:消费方测试可能因断言失败带锁 panic
            self.log.lock().unwrap_or_else(|e| e.into_inner()).push(ev);
        }
    }

    impl AsrSession for ProbeSession {
        fn push(&mut self, _pcm: &[i16], out: &mut dyn AsrEvents) {
            if self.dead {
                return;
            }
            self.record("push");
            if !out.emit(AsrEvent::Partial(AsrText::plain("x"))) {
                self.dead = true;
            }
        }

        fn finish(&mut self, out: &mut dyn AsrEvents) {
            self.record("finish");
            let _ = out.emit(AsrEvent::End(AsrText::plain("end")));
        }
    }

    impl Drop for ProbeSession {
        fn drop(&mut self) {
            self.record("drop");
        }
    }

    /// 消费端关闭:worker 停止驱动、丢弃会话,不调用 finish
    /// (与旧版 blocking_send 出错即 return 等价)
    #[test]
    fn consumer_close_drops_session_without_finish() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (handle, rx) = start_with(
            Box::new(ProbeBackend(log.clone())),
            Config::default(),
            AudioIn::MONO_16K,
        );
        handle.push(&[0i16; 1600]);
        drop(rx);
        for _ in 0..8 {
            handle.push(&[0i16; 1600]); // 至少一块的 emit 必然失败
        }
        handle.finish(); // End 不再被处理:worker 在此之前已停止驱动

        // 读快照(不持锁断言),等 worker 退出(会话 drop)
        let snapshot = |log: &Arc<Mutex<Vec<&'static str>>>| {
            log.lock().unwrap_or_else(|e| e.into_inner()).clone()
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        while !snapshot(&log).contains(&"drop") {
            assert!(
                Instant::now() < deadline,
                "worker 未退出: {:?}",
                snapshot(&log)
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let l = snapshot(&log);
        assert!(l.contains(&"push"), "至少应处理过一块: {l:?}");
        assert!(!l.contains(&"finish"), "消费端关闭后不应定稿: {l:?}");
    }

    struct ProbeBackend(Arc<Mutex<Vec<&'static str>>>);

    impl AsrBackend for ProbeBackend {
        fn open(&self, _cfg: &Config, _audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>> {
            Ok(Box::new(ProbeSession {
                log: self.0.clone(),
                dead: false,
            }))
        }
    }
}
