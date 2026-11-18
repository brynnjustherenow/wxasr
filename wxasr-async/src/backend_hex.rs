//! hex-asr 后端:GGUF 推理栈(transcribe-cpp,parakeet/whisper/Qwen 等
//! 模型家族)接入统一层。
//!
//! [`HexBackend`] 持有一个已加载的 [`hex_asr::Engine`](共享的专用推理
//! 线程,模型常驻,多次会话零重复加载);每次 [`open`](AsrBackend::open)
//! 按模型能力分流:能真流式 → [`HexStream`](流式轮换会话);不能且
//! 伪流式开启 → [`PseudoSession`];否则原行为(离线单段,只出 `End`)。
//! `Config::partial` 控制是否发中间结果;`partial_min_fresh` 忽略
//! (hex 的流自身带节流);`Config::rotate` 由本后端解释(见下)。
//!
//! # 上下文轮换
//!
//! native 流是一次性的:`finalize()` 是终态,没有"剥离左文/增量定稿"
//! API。轮换因此 = **老流 finish + 新流从零开始**:finish 排空 ≤~2.1s
//! 的解码滞后尾巴(落在**本次** Commit 里),之后的新音频进新流,零重叠
//! 零丢失;识别线程阻塞期间统一层通道排队,不丢音。
//!
//! 轮换产生 [`Commit`](crate::AsrEvent::Commit)(段文本,会话继续);
//! [`finish`](AsrSession::finish) 定稿最后一段为 [`End`](crate::AsrEvent::End)。
//! 判定与 wxasr 后端共用 [`RotatePolicy`](crate::policy::RotatePolicy):
//!
//! - `stream_ms`:当前流喂入的音频毫秒;
//! - `silence_ms`:尾随静音——Silero VAD 流式概率([`HexBackend::with_vad`],
//!   缺失或运行失败时退化为能量门,行为同 backend_wxasr 的轻量 VAD),
//!   轮换**不清零**(静音描述音频本身);
//! - `punct_ms`:committed 文本新增句末标点时的 `stream_ms`。committed
//!   即流式稳定前缀,比音频滞后 ≤~1s(解码延迟)——`Commit` 因此可能
//!   含标点之后的几个滞后词。hex 无 `rotate_at_punct` 等价物,切点 =
//!   判定时刻(而非标点位置),与 wxasr 后端的差异在此。
//!
//! 未开伪流式时,非流式模型(whisper/qwen)committed 永不增长,Punct
//! 永不触发,只有 `hard_ms` 起效——每段整段转写(fallback 天然退化成
//! 按段二遍)。[`RotatePolicy::Never`] 即旧行为:单段、finish 只出 `End`。
//!
//! # 伪流式(离线模型的完整事件行为)
//!
//! whisper/qwen 等离线模型没有 native 流(`Engine::begin()` → `None`),
//! 原生事件行为只剩 finish 时一个 `End`。伪流式一族(whisper_streaming/
//! RealtimeSTT 同族思路)在**后端层**补全:增长缓冲 + 周期性整段重转写
//! → 增长(可回改)文本当 [`Partial`](契约里 Partial 是投机显示,整体
//! 回改合法);轮换判定触发时做一次段权威转写 → `Commit`,继续收音;
//! `finish` → 最后一段 `End`。见 [`HexBackend::pseudo_streaming`] 与
//! [`PseudoSession`]。模型能真流式时自动升级真流式路径(拿 parakeet
//! 跑伪流式后端不退化);整段转写的吞吐决定 partial 节奏与 Commit
//! 后向延迟(段长/RTF,伪流式的固有代价),阻塞期间音频排队不丢。
//!
//! 富字段直通:[`AsrText::language`] 来自模型自报的语言识别
//! (whisper 等),词/段时间戳字段在引擎以时间戳级别运行时填充
//! (默认整段路径不带时间戳,字段为空,见
//! [`hex_asr::Transcript`])。

use std::sync::Arc;

use crate::policy::{RotateCtx, RotatePolicy};
use crate::preprocess::InputConverter;
use crate::{
    AsrBackend, AsrEvent, AsrEvents, AsrSegment, AsrSession, AsrText, AsrWord, AudioIn, Config,
};
use hex_asr::{Engine, ModelSpec, SessionEvent, SileroVad, StreamingSession, Transcript};

/// 句末标点(与 policy 的 `Punct` 语义一致:切在句界,Commit 以完整句
/// 收尾;逗号/顿号不是句界,不触发)
const SENTENCE_END: &[char] = &['.', '!', '?', '。', '！', '？'];

/// hex-asr 后端:封装一个已加载的引擎。
///
/// 引擎的加载(含模型加载与设备选择)是重操作,由调用方完成并共享:
/// `HexBackend::new(Arc<Engine>)`;或用 [`HexBackend::from_spec`] 一步
/// 加载。多个并发会话共享同一引擎(引擎内部按序调度)。
pub struct HexBackend {
    engine: Arc<Engine>,
    /// 常驻的 Silero VAD 模型;每个会话 fork 独立实例(流位置与静音
    /// 累计是会话级状态,不能串扰)。`None` → 会话用能量门。VAD 只
    /// `Send` 不 `Sync`,包 Mutex 让后端满足 `Sync`(锁只在 fork 时持有)
    vad: Option<std::sync::Mutex<SileroVad>>,
    /// `Some(partial_every_ms)` = 伪流式开启:模型在此设备不能真流式
    /// 时会话走 [`PseudoSession`](能真流式则升级真流式路径,不退化)。
    /// `None` = 原行为。
    pseudo: Option<u32>,
}

/// 伪流式专属参数([`HexBackend::with_pseudo`] 的后端构造参数,不进
/// `Config`——契约:后端专属配置走构造函数)。
pub struct PseudoOptions {
    /// 两次 partial 重转写的最小音频间隔(毫秒,按喂入的音频时间
    /// 计,非墙钟)。默认 2000:GPU 上 20s 段整段转写 ≈1s,2s 节奏的
    /// 稳态 GPU 占用 ~50-60%。
    pub partial_every_ms: u32,
    /// Silero VAD(静音 endpointing 信号);缺省退化能量门(同流式
    /// 路径)。
    pub vad: Option<SileroVad>,
}

impl Default for PseudoOptions {
    fn default() -> Self {
        PseudoOptions {
            partial_every_ms: 2_000,
            vad: None,
        }
    }
}

impl HexBackend {
    /// 用已加载的引擎创建(推荐:调用方控制加载时机与设备)。
    pub fn new(engine: Arc<Engine>) -> Self {
        HexBackend {
            engine,
            vad: None,
            pseudo: None,
        }
    }

    /// 用已加载的引擎 + 流式 Silero VAD 创建:静音信号走 DNN 概率
    /// 而非能量门(`Auto` 族轮换策略的 endpointing 质量)。
    pub fn with_vad(engine: Arc<Engine>, vad: SileroVad) -> Self {
        HexBackend {
            engine,
            vad: Some(std::sync::Mutex::new(vad)),
            pseudo: None,
        }
    }

    /// 伪流式:离线模型(whisper/qwen)也获得完整事件行为——周期性
    /// 整段重转写产生可回改的 `Partial`,轮换切段定稿 `Commit`,
    /// `finish` 出 `End`。等价
    /// `with_pseudo(engine, PseudoOptions::default())`。模型能真流式
    /// 时自动升级真流式路径(拿 parakeet 跑也不退化)。
    pub fn pseudo_streaming(engine: Arc<Engine>) -> Self {
        Self::with_pseudo(engine, PseudoOptions::default())
    }

    /// [`Self::pseudo_streaming`] 的完整选项(partial 节奏与 VAD)。
    pub fn with_pseudo(engine: Arc<Engine>, opts: PseudoOptions) -> Self {
        HexBackend {
            engine,
            vad: opts.vad.map(std::sync::Mutex::new),
            pseudo: Some(opts.partial_every_ms),
        }
    }

    /// 加载并创建。`Engine::load` 的直通(模型加载、设备选择、预热)。
    pub fn from_spec(spec: &ModelSpec, device: hex_asr::Device) -> eyre::Result<Self> {
        Ok(HexBackend {
            engine: Arc::new(Engine::load(spec, device)?),
            vad: None,
            pseudo: None,
        })
    }

    /// [`Self::with_vad`] 与 [`Self::from_spec`] 的组合。
    pub fn from_spec_with_vad(
        spec: &ModelSpec,
        device: hex_asr::Device,
        vad: SileroVad,
    ) -> eyre::Result<Self> {
        Ok(HexBackend {
            engine: Arc::new(Engine::load(spec, device)?),
            vad: Some(std::sync::Mutex::new(vad)),
            pseudo: None,
        })
    }
}

impl AsrBackend for HexBackend {
    fn open(&self, cfg: &Config, audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>> {
        // hex-asr 的会话吃 16 kHz 单声道 i16:调用方声明的格式在这里转换,
        // 16 kHz 单声道直通。
        let convert = InputConverter::new(audio)
            .map_err(|reason| eyre::eyre!("hex-asr 不支持该音频格式: {reason}"))?;
        let vad = match &self.vad {
            Some(vad) => {
                // poison 不可能(持锁内不 panic),容忍是为了健壮
                let held = vad.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                Some(
                    held.fork()
                        .map_err(|error| eyre::eyre!("Silero VAD 会话创建失败: {error:#}"))?,
                )
            }
            None => None,
        };
        let silence = vad.map_or(SilenceSource::default(), SilenceSource::Vad);
        // 伪流式仅在模型此设备不能真流式时生效;能流式则自动升级
        // (伪流式后端拿 parakeet 也不退化)
        if let Some(partial_every_ms) = self.pseudo
            && !self.engine.streams()
        {
            return Ok(Box::new(PseudoSession::new(
                self.engine.clone(),
                convert,
                silence,
                cfg.rotate.clone(),
                cfg.partial,
                partial_every_ms,
            )));
        }
        Ok(Box::new(HexStream {
            engine: self.engine.clone(),
            session: StreamingSession::open(self.engine.clone(), cfg.partial),
            convert,
            partial: cfg.partial,
            rotate: cfg.rotate.clone(),
            silence,
            f32: Vec::new(),
            stream_ms: 0,
            session_ms: 0,
            punct_ms: None,
            committed_len: 0,
        }))
    }
}

/// hex-asr 会话:一个 [`StreamingSession`] + 输入格式转换 + 轮换判定。
pub struct HexStream {
    /// 轮换时开新流用(共享引擎,内部按序调度)
    engine: Arc<Engine>,
    session: StreamingSession,
    convert: InputConverter,
    /// 开新流时透传给 `StreamingSession::open`
    partial: bool,
    rotate: RotatePolicy,
    /// 静音信号源(VAD 或能量门),轮换不清零
    silence: SilenceSource,
    /// i16 → f32 转换 scratch(喂 VAD 用)
    f32: Vec<f32>,
    /// 当前流累计音频(毫秒)
    stream_ms: u32,
    /// 自会话起点累计的音频毫秒(轮换不清零;Commit/End 时间戳平移到
    /// 会话时间轴用,契约:相对会话起点)
    session_ms: u32,
    /// committed 文本最近一次出现句末标点的位置;无则 `None`
    punct_ms: Option<u32>,
    /// committed 文本长度游标(字节;committed 在一个流内只增,游标单调)
    committed_len: usize,
}

impl HexStream {
    /// 喂一块 16 kHz 单声道 i16:静音信号与识别会话并行推进,更新计时
    /// 与标点游标,再按策略判定轮换。
    fn drive(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
        let ms = pcm.len() as u32 / 16; // 16000 Hz → 16 样本/ms

        // VAD 吃 f32;会话内部自己做 i16→f32。先后无所谓(互不依赖)
        self.f32.clear();
        hex_asr::audio::i16_to_f32(pcm, &mut self.f32);
        let (silence_ms, _) = self.silence.feed(pcm, &self.f32);
        self.session.push(
            pcm,
            &mut Bridge {
                out,
                commit: false,
                offset: 0,
            },
        );
        self.stream_ms += ms;
        self.session_ms = self.session_ms.saturating_add(ms);

        // committed(流式稳定前缀)新增部分出现句末标点 → 记 punct_ms
        // (标点出现时刻;比音频里的真实句界滞后 ≤~1s 解码延迟)
        if new_sentence_end(self.session.committed_text(), &mut self.committed_len) {
            self.punct_ms = Some(self.stream_ms);
        }

        if self.stream_ms > 0
            && self.rotate.should_rotate(&RotateCtx {
                stream_ms: self.stream_ms,
                silence_ms,
                punct_ms: self.punct_ms,
            })
        {
            self.rotate_stream(out);
        }
    }

    /// 当前流在会话时间轴上的起点(毫秒):累计喂入 − 当前流已喂入。
    fn stream_offset_ms(&self) -> i64 {
        i64::from(self.session_ms.saturating_sub(self.stream_ms))
    }

    /// 定稿当前段([`AsrEvent::Commit`])并开新流:native 流一次性,
    /// 轮换 = 老流 finish 排空滞后尾巴 + 新流从零。VAD 静音不清零
    /// (静音描述音频本身,policy 语义,同 backend_wxasr)。
    fn rotate_stream(&mut self, out: &mut dyn AsrEvents) {
        let offset = self.stream_offset_ms();
        self.session.finish(&mut Bridge {
            out,
            commit: true,
            offset,
        });
        self.session = StreamingSession::open(self.engine.clone(), self.partial);
        self.stream_ms = 0;
        self.punct_ms = None;
        self.committed_len = 0;
    }
}

impl AsrSession for HexStream {
    fn push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
        let converted = self.convert.push(pcm);
        if converted.is_empty() {
            return;
        }
        self.drive(&converted, out);
    }

    fn finish(&mut self, out: &mut dyn AsrEvents) {
        // 重采样残留先进会话(不再参与轮换判定),再定稿最后一段
        let tail = self.convert.flush();
        if !tail.is_empty() {
            self.session.push(
                &tail,
                &mut Bridge {
                    out,
                    commit: false,
                    offset: 0,
                },
            );
        }
        let offset = self.stream_offset_ms();
        self.session.finish(&mut Bridge {
            out,
            commit: false,
            offset,
        }); // → End
    }
}

/// [`PseudoSession`] 的"转写一段":输入当前段缓冲,返回整段转写结果
/// (`None` = 转写失败,放弃本段文本)。真实现走引擎整段转写;单测
/// 注入假转写器(按段内容/长度生成文本)。
type SegmentTranscriber = Box<dyn FnMut(&[f32]) -> Option<Transcript> + Send>;

/// 伪流式会话:离线模型(无 native 流)的完整事件行为(模块文档
/// "伪流式"一节)。增长缓冲 + 周期性整段重转写,状态机全在本层,
/// 引擎只看到一个又一个整段转写 job。
///
/// - `Partial`:距上次转写 ≥ `partial_every_ms` 且本段见过语音时重转写
///   当前段,文本有变即发——**允许整体回改**(伪流式的本质;契约里
///   Partial 是投机显示用)。全静音段跳过重转写(省 GPU)。
/// - `Commit`:轮换判定(共用 [`RotatePolicy`],`punct_ms` 在最近一次
///   转写文本上近似)触发时做一次段权威转写,非空即发;段缓冲清零
///   (VAD 静音不清零,policy 语义)。
/// - `End`:`finish` 对残留缓冲做最后一次权威转写,非空即发。
///
/// 整段转写阻塞识别线程期间,后续 push 在统一层通道排队不丢音;
/// GPU 上 20s 段一次转写 ≈1s(whisper-turbo 19× 实时),`hard_ms`
/// (默认 20s)即段缓冲的自然上界。转写失败只 warn 一次并放弃该段
/// 文本(CPU+whisper 会超引擎 job deadline,属设计内快速失败)。
pub struct PseudoSession {
    /// 一段 → 转写结果;真实现捕获 `Arc<Engine>` 整段转写
    transcribe: SegmentTranscriber,
    convert: InputConverter,
    /// 当前段缓冲(16 kHz 单声道 f32,轮换清零)
    segment: Vec<f32>,
    /// 静音信号源(VAD 或能量门),轮换不清零
    silence: SilenceSource,
    /// 本段是否出现过语音(全静音不重转写)
    speech_seen: bool,
    /// 当前段累计音频毫秒
    stream_ms: u32,
    /// 距上次转写(权威或重转写)的音频毫秒(partial 节流钟)
    since_transcribe_ms: u32,
    /// 自会话起点累计的音频毫秒(轮换不清零;Commit/End 时间戳平移到
    /// 会话时间轴用,契约:相对会话起点)
    session_ms: u32,
    /// 最近一次转写文本新增句末标点时的 `stream_ms`;无则 `None`
    punct_ms: Option<u32>,
    /// 最近一次转写文本(partial 去重基准 + 标点近似的载体;可回改)
    last_text: String,
    /// `last_text` 的扫描游标(字节;文本可回改,回缩时 get 安全截断)
    punct_len: usize,
    rotate: RotatePolicy,
    partial: bool,
    partial_every_ms: u32,
    /// 转写失败已告警过(只 warn 一次)
    warned_fail: bool,
    /// 消费端已关闭:停止一切计算
    dead: bool,
    /// i16 → f32 转换 scratch(喂 VAD / 入段缓冲)
    f32: Vec<f32>,
}

impl PseudoSession {
    /// 真实现的"转写一段":引擎整段转写(dictation 级优先级,引擎内部
    /// 按序调度;含模型的 padding/trailing silence 处理)。
    fn new(
        engine: Arc<Engine>,
        convert: InputConverter,
        silence: SilenceSource,
        rotate: RotatePolicy,
        partial: bool,
        partial_every_ms: u32,
    ) -> Self {
        let transcribe: SegmentTranscriber = Box::new(move |segment| {
            engine
                .transcribe_clip(segment.to_vec())
                .ok()?
                .wait()
                .ok()
                .map(|transcribed| transcribed.transcript)
        });
        Self::with_transcriber(
            transcribe,
            convert,
            silence,
            rotate,
            partial,
            partial_every_ms,
        )
    }

    /// 单测注入口:假转写器驱动整个状态机。
    fn with_transcriber(
        transcribe: SegmentTranscriber,
        convert: InputConverter,
        silence: SilenceSource,
        rotate: RotatePolicy,
        partial: bool,
        partial_every_ms: u32,
    ) -> Self {
        PseudoSession {
            transcribe,
            convert,
            segment: Vec::new(),
            silence,
            speech_seen: false,
            stream_ms: 0,
            since_transcribe_ms: 0,
            session_ms: 0,
            punct_ms: None,
            last_text: String::new(),
            punct_len: 0,
            rotate,
            partial,
            partial_every_ms,
            warned_fail: false,
            dead: false,
            f32: Vec::new(),
        }
    }

    /// 喂一块 16 kHz 单声道 i16:静音信号与段缓冲并行推进,更新计时
    /// 与标点近似,再按策略判定轮换 / partial 节流。
    fn drive(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
        let ms = pcm.len() as u32 / 16; // 16000 Hz → 16 样本/ms

        self.f32.clear();
        hex_asr::audio::i16_to_f32(pcm, &mut self.f32);
        let (silence_ms, is_speech) = self.silence.feed(pcm, &self.f32);
        self.speech_seen |= is_speech;
        self.stream_ms = self.stream_ms.saturating_add(ms);
        self.session_ms = self.session_ms.saturating_add(ms);
        self.since_transcribe_ms = self.since_transcribe_ms.saturating_add(ms);
        self.segment.extend_from_slice(&self.f32);

        if self.stream_ms > 0
            && self.rotate.should_rotate(&RotateCtx {
                stream_ms: self.stream_ms,
                silence_ms,
                punct_ms: self.punct_ms,
            })
        {
            self.commit_segment(out);
            // 段状态清零;静音不清零(描述音频本身)。转写钟一并归零
            // (权威转写刚做过)
            self.segment.clear();
            self.speech_seen = false;
            self.stream_ms = 0;
            self.since_transcribe_ms = 0;
            self.punct_ms = None;
            self.last_text.clear();
            self.punct_len = 0;
            return;
        }

        if !self.partial
            || !self.speech_seen
            || self.since_transcribe_ms < self.partial_every_ms
            || self.segment.is_empty()
        {
            return;
        }
        match (self.transcribe)(&self.segment) {
            Some(transcript) => {
                // 整体回改合法:只按"与上次文本不同"去重;标点在最新
                // 文本的新增部分上近似(文本回缩时游标安全截断)
                let changed = transcript.text != self.last_text;
                let punct_hit = new_sentence_end(&transcript.text, &mut self.punct_len);
                self.last_text.clone_from(&transcript.text);
                if punct_hit {
                    self.punct_ms = Some(self.stream_ms);
                }
                self.since_transcribe_ms = 0;
                if changed
                    && !self.last_text.trim().is_empty()
                    && !out.emit(AsrEvent::Partial(asr_text(transcript)))
                {
                    self.close();
                }
            }
            None => self.warn_transcribe_failed(),
        }
    }

    /// 当前段在会话时间轴上的起点(毫秒):累计喂入 − 当前段已喂入
    /// (finish 时重采样残留只加长段尾,不动段起点,公式不变)。
    fn segment_offset_ms(&self) -> i64 {
        i64::from(self.session_ms.saturating_sub(self.stream_ms))
    }

    /// 段权威转写 → [`AsrEvent::Commit`](空文本不发;消费端关闭即停)。
    /// 时间戳平移到会话时间轴(契约:相对会话起点)。
    fn commit_segment(&mut self, out: &mut dyn AsrEvents) {
        if self.segment.is_empty() {
            return;
        }
        let offset = self.segment_offset_ms();
        match (self.transcribe)(&self.segment) {
            Some(mut transcript) => {
                offset_transcript(&mut transcript, offset);
                if !transcript.text.trim().is_empty()
                    && !out.emit(AsrEvent::Commit(asr_text(transcript)))
                {
                    self.close();
                }
            }
            None => self.warn_transcribe_failed(),
        }
    }

    /// 转写失败只告警一次(连续失败多为设备不适合,如 CPU+whisper 超
    /// 引擎 deadline;后续失败静默放弃该段文本)。
    fn warn_transcribe_failed(&mut self) {
        if !self.warned_fail {
            self.warned_fail = true;
            tracing::warn!(
                "伪流式整段转写失败(如 CPU+whisper 会超引擎 job deadline,属设计内快速失败);后续失败不再重复告警"
            );
        }
    }

    /// 消费端已关闭:停止计算,释放段缓冲。
    fn close(&mut self) {
        self.dead = true;
        self.segment.clear();
    }
}

impl AsrSession for PseudoSession {
    fn push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
        if self.dead {
            return;
        }
        let converted = self.convert.push(pcm);
        if converted.is_empty() {
            return;
        }
        self.drive(&converted, out);
    }

    fn finish(&mut self, out: &mut dyn AsrEvents) {
        if self.dead {
            return;
        }
        self.dead = true;
        // 重采样残留并入段缓冲(不再参与轮换判定),最后一次权威转写
        let tail = self.convert.flush();
        if !tail.is_empty() {
            self.f32.clear();
            hex_asr::audio::i16_to_f32(&tail, &mut self.f32);
            self.segment.extend_from_slice(&self.f32);
        }
        if self.segment.is_empty() {
            return;
        }
        let offset = self.segment_offset_ms();
        match (self.transcribe)(&self.segment) {
            Some(mut transcript) => {
                offset_transcript(&mut transcript, offset);
                if !transcript.text.trim().is_empty() {
                    out.emit(AsrEvent::End(asr_text(transcript)));
                }
            }
            None => self.warn_transcribe_failed(),
        }
    }
}

/// 静音信号源:Silero DNN 帧概率;缺失或运行失败时退化能量门
/// (复用 backend_wxasr 的轻量 VAD,行为一致)。两者都只在会话内累计,
/// 轮换不清零。
enum SilenceSource {
    /// Silero:概率 ≥0.5 语音,<0.35 静音(32ms/帧累计)
    Vad(SileroVad),
    /// 能量门(RMS 峰值自适应噪声底)
    Energy {
        gate: crate::backend_wxasr::Vad,
        quiet_ms: u32,
    },
}

impl Default for SilenceSource {
    fn default() -> Self {
        SilenceSource::Energy {
            gate: crate::backend_wxasr::Vad::default(),
            quiet_ms: 0,
        }
    }
}

impl SilenceSource {
    /// 喂一块 16 kHz 单声道音频(i16 及其 f32 视图),返回(当前连续
    /// 静音毫秒,本块是否语音)。能量门的 `is_speech` 有状态(自适应
    /// 噪声底),一次 feed 只判一次,调用方别再单独问。VAD 喂失败
    /// (坏样本/会话损坏)时降级能量门并立即起算。
    fn feed(&mut self, pcm: &[i16], f32: &[f32]) -> (u32, bool) {
        let ms = pcm.len() as u32 / 16;
        match self {
            SilenceSource::Vad(vad) => match vad.feed(f32) {
                Ok(()) => (vad.silence_ms(), vad.is_speech()),
                Err(error) => {
                    tracing::warn!(%error, "Silero VAD 失败,静音信号退化为能量门");
                    let mut gate = crate::backend_wxasr::Vad::default();
                    let speech = gate.is_speech(pcm);
                    let quiet_ms = if speech { 0 } else { ms };
                    *self = SilenceSource::Energy { gate, quiet_ms };
                    (quiet_ms, speech)
                }
            },
            SilenceSource::Energy { gate, quiet_ms } => {
                let speech = gate.is_speech(pcm);
                *quiet_ms = if speech {
                    0
                } else {
                    quiet_ms.saturating_add(ms)
                };
                (*quiet_ms, speech)
            }
        }
    }
}

/// 扫描 committed 文本自游标起的新增部分是否含句末标点;游标推到文本
/// 末尾(committed 在一个流内只增,游标单调;轮换时归零)。命中返回
/// `true`——同一处标点只报一次。
fn new_sentence_end(committed: &str, cursor: &mut usize) -> bool {
    let fresh = committed.get(*cursor..).unwrap_or("");
    let hit = fresh.contains(SENTENCE_END);
    *cursor = committed.len();
    hit
}

/// hex-asr 的 [`SessionEvent`] → 统一层的 [`AsrEvent`]。`Final` 只在
/// session finish 时出现:轮换定稿映射成 `Commit`,会话结束映射成 `End`。
/// 时间戳按 `offset` 平移到会话时间轴(转写结果相对本次流起点)。
struct Bridge<'a> {
    out: &'a mut dyn AsrEvents,
    /// 本次 finish 的定稿映射:轮换 → Commit;会话结束 → End
    commit: bool,
    /// 本次流起点的会话偏移(毫秒);Partial 无时间戳,忽略
    offset: i64,
}

impl hex_asr::SessionEvents for Bridge<'_> {
    fn emit(&mut self, event: SessionEvent) -> bool {
        match event {
            SessionEvent::Partial(text) => self.out.emit(AsrEvent::Partial(AsrText::plain(text))),
            SessionEvent::Final(mut transcript) => {
                offset_transcript(&mut transcript, self.offset);
                let text = asr_text(transcript);
                if self.commit {
                    self.out.emit(AsrEvent::Commit(text))
                } else {
                    self.out.emit(AsrEvent::End(text))
                }
            }
        }
    }
}

/// 段/流的转写时间戳相对本次转写起点;契约要求相对**会话**起点
/// (`AsrSegment`/`AsrWord` 文档)——Commit/End 前统一平移。
fn offset_transcript(transcript: &mut Transcript, offset_ms: i64) {
    for segment in &mut transcript.segments {
        segment.start_ms += offset_ms;
        segment.end_ms += offset_ms;
    }
    for word in &mut transcript.words {
        word.start_ms += offset_ms;
        word.end_ms += offset_ms;
    }
}

/// [`hex_asr::Transcript`] → [`AsrText`]:富字段直通,时间戳毫秒原样。
fn asr_text(transcript: Transcript) -> AsrText {
    // 整段说话人:所有分段归属同一个说话人时才有,否则 None
    let speaker = transcript
        .segments
        .iter()
        .map(|segment| segment.speaker)
        .reduce(|acc, speaker| match (acc, speaker) {
            (Some(a), Some(b)) if a == b => Some(a),
            _ => None,
        })
        .flatten();
    AsrText {
        content: transcript.text,
        language: transcript.language,
        speaker,
        segments: transcript
            .segments
            .into_iter()
            .map(|segment| AsrSegment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text,
                speaker: segment.speaker,
            })
            .collect(),
        words: transcript
            .words
            .into_iter()
            .map(|word| AsrWord {
                start_ms: word.start_ms,
                end_ms: word.end_ms,
                text: word.text,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sentence_end_in_new_text_hits_once() {
        let mut cursor = 0;
        assert!(!new_sentence_end("", &mut cursor));
        assert!(!new_sentence_end("hello world", &mut cursor), "无标点");
        assert!(new_sentence_end("hello world.", &mut cursor), "新增句号");
        assert!(
            !new_sentence_end("hello world. more", &mut cursor),
            "旧标点不重复报"
        );
        assert!(new_sentence_end("hello world. more?", &mut cursor));
        assert_eq!(cursor, "hello world. more?".len());
    }

    #[test]
    fn fullwidth_sentence_end_also_hits() {
        let mut cursor = 0;
        assert!(!new_sentence_end("你好", &mut cursor));
        assert!(new_sentence_end("你好。", &mut cursor), "全角句号");
        let mut cursor = 0;
        assert!(new_sentence_end("走吗?", &mut cursor), "半角问号");
        let mut cursor = 0;
        assert!(new_sentence_end("是嗎?", &mut cursor), "全角问号");
    }

    #[test]
    fn a_shrinking_text_never_panics() {
        // 一个流内 committed 只增,但游标越界(轮换后复用失误)也应安全
        let mut cursor = 10;
        assert!(!new_sentence_end("hi.", &mut cursor));
        assert_eq!(cursor, 3);
    }

    #[test]
    fn the_energy_gate_accumulates_silence_and_clears_on_speech() {
        let mut source = SilenceSource::default();
        let silence = vec![0i16; 1600]; // 100 ms
        let f32 = |pcm: &[i16]| -> Vec<f32> {
            let mut out = Vec::new();
            hex_asr::audio::i16_to_f32(pcm, &mut out);
            out
        };
        assert_eq!(source.feed(&silence, &f32(&silence)), (100, false));
        assert_eq!(source.feed(&silence, &f32(&silence)), (200, false));
        // 440Hz 中等音量:语音,清零(backend_wxasr::Vad 已单测其判定)
        let tone: Vec<i16> = (0..1600)
            .map(|i| {
                (3000.0 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin()) as i16
            })
            .collect();
        assert_eq!(source.feed(&tone, &f32(&tone)), (0, true));
        assert_eq!(source.feed(&silence, &f32(&silence)), (100, false));
    }

    #[test]
    fn a_timed_transcript_maps_onto_the_rich_event() {
        let transcript = Transcript {
            text: "hello world".into(),
            language: Some("en".into()),
            segments: vec![
                hex_asr::TranscriptSegment {
                    start_ms: 0,
                    end_ms: 500,
                    text: "hello".into(),
                    speaker: Some(1),
                },
                hex_asr::TranscriptSegment {
                    start_ms: 500,
                    end_ms: 900,
                    text: "world".into(),
                    speaker: Some(1),
                },
            ],
            words: vec![hex_asr::TranscriptSegment {
                start_ms: 0,
                end_ms: 300,
                text: "hello".into(),
                speaker: None,
            }],
        };
        let text = asr_text(transcript);
        assert_eq!(text.content, "hello world");
        assert_eq!(text.language.as_deref(), Some("en"));
        assert_eq!(text.speaker, Some(1), "统一归属的说话人上浮到整段");
        assert_eq!(text.segments.len(), 2);
        assert_eq!(text.segments[0].speaker, Some(1));
        assert_eq!(text.words.len(), 1);
        assert_eq!(text.words[0].end_ms, 300);
    }

    #[test]
    fn mixed_speakers_do_not_surface_a_span_speaker() {
        let transcript = Transcript {
            text: "a b".into(),
            language: None,
            segments: vec![
                hex_asr::TranscriptSegment {
                    start_ms: 0,
                    end_ms: 100,
                    text: "a".into(),
                    speaker: Some(1),
                },
                hex_asr::TranscriptSegment {
                    start_ms: 100,
                    end_ms: 200,
                    text: "b".into(),
                    speaker: Some(2),
                },
            ],
            words: Vec::new(),
        };
        let text = asr_text(transcript);
        assert_eq!(text.speaker, None, "混合说话人不上浮");
        assert_eq!(text.segments[0].speaker, Some(1));
        assert_eq!(text.segments[1].speaker, Some(2));
    }

    // ── PseudoSession:假转写器驱动整个状态机 ────────────────

    mod pseudo {
        use super::*;

        /// 记录事件的 sink;`alive: false` 模拟消费端已关闭
        struct Sink {
            events: Vec<AsrEvent>,
            alive: bool,
        }

        impl AsrEvents for Sink {
            fn emit(&mut self, event: AsrEvent) -> bool {
                self.events.push(event);
                self.alive
            }
        }

        /// 440Hz 中等音量(与能量门测试同款,判为语音)
        fn speech_ms(ms: u32) -> Vec<i16> {
            (0..ms as usize * 16)
                .map(|i| {
                    (3000.0 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin())
                        as i16
                })
                .collect()
        }

        fn silence_ms(ms: u32) -> Vec<i16> {
            vec![0_i16; ms as usize * 16]
        }

        fn text_transcript(text: String) -> Transcript {
            Transcript {
                text,
                language: None,
                segments: Vec::new(),
                words: Vec::new(),
            }
        }

        /// 契约:Commit/End 时间戳相对**会话**起点——第二个 Commit 的段
        /// 从第一段结束处起算,而不是每段从 0 重来(轮换平移)。
        #[test]
        fn commit_timestamps_are_session_absolute() {
            let transcribe: SegmentTranscriber = Box::new(|segment: &[f32]| {
                let ms = segment.len() as i64 / 16;
                Some(Transcript {
                    text: format!("{ms}ms"),
                    language: None,
                    segments: vec![hex_asr::TranscriptSegment {
                        start_ms: 0,
                        end_ms: ms,
                        text: String::new(),
                        speaker: None,
                    }],
                    words: Vec::new(),
                })
            });
            // Custom:恰好 2s 一换 → 4.5s 喂入 = Commit(0–2s)+Commit(2–4s)+End(4–4.5s)
            let rotate = RotatePolicy::Custom(Arc::new(|ctx: &RotateCtx| ctx.stream_ms >= 2_000));
            let mut session = pseudo(rotate, u32::MAX, transcribe);
            let mut sink = Sink {
                events: Vec::new(),
                alive: true,
            };
            for _ in 0..225 {
                session.push(&speech_ms(20), &mut sink);
            }
            session.finish(&mut sink);
            let spans: Vec<(i64, i64)> = sink
                .events
                .iter()
                .filter_map(|event| match event {
                    AsrEvent::Commit(text) | AsrEvent::End(text) => {
                        text.segments.first().map(|s| (s.start_ms, s.end_ms))
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(spans, vec![(0, 2_000), (2_000, 4_000), (4_000, 4_500)]);
        }

        fn pseudo(
            rotate: RotatePolicy,
            partial_every_ms: u32,
            transcribe: SegmentTranscriber,
        ) -> PseudoSession {
            PseudoSession::with_transcriber(
                transcribe,
                InputConverter::new(AudioIn::MONO_16K).expect("16k mono 直通"),
                SilenceSource::default(),
                rotate,
                true,
                partial_every_ms,
            )
        }

        /// 按段长度生成文本的假转写器,记录每次调用的段长(样本数)
        fn len_transcriber(calls: Arc<std::sync::Mutex<Vec<usize>>>) -> SegmentTranscriber {
            Box::new(move |segment: &[f32]| {
                calls.lock().unwrap().push(segment.len());
                Some(text_transcript(format!("{}ms", segment.len() / 16)))
            })
        }

        /// 事件文本按序(只看 Partial/Commit/End 的 content)
        fn contents(events: &[AsrEvent]) -> Vec<&str> {
            events
                .iter()
                .map(|event| match event {
                    AsrEvent::Partial(text) | AsrEvent::Commit(text) | AsrEvent::End(text) => {
                        text.content.as_str()
                    }
                })
                .collect()
        }

        /// 事件种类按序(P/C/E)
        fn kinds(events: &[AsrEvent]) -> Vec<&'static str> {
            events
                .iter()
                .map(|event| match event {
                    AsrEvent::Partial(_) => "P",
                    AsrEvent::Commit(_) => "C",
                    AsrEvent::End(_) => "E",
                })
                .collect()
        }

        /// 能量门的噪声底由首个块播种:先喂 200ms 静音(真实采集以
        /// 环境音开场),之后的语音块才判为语音。
        fn ambient_start(session: &mut PseudoSession, sink: &mut Sink) {
            session.push(&silence_ms(200), sink);
        }

        /// 事件序 `(Partial* Commit)* End?`:partial 按节流间隔出现,
        /// 文本随段增长(允许回改,这里只增长);finish 权威转写一段 End。
        #[test]
        fn partial_retranscription_is_throttled_and_finish_ends() {
            let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut session = pseudo(RotatePolicy::Never, 2_000, len_transcriber(calls.clone()));
            let mut sink = Sink {
                events: Vec::new(),
                alive: true,
            };
            ambient_start(&mut session, &mut sink);
            for _ in 0..10 {
                session.push(&speech_ms(1_000), &mut sink); // 共 10s 语音
            }
            session.finish(&mut sink);
            // 重转写自 since ≥ 2s 起每 2s 一次,段长 2200..10200ms
            assert_eq!(
                contents(&sink.events),
                ["2200ms", "4200ms", "6200ms", "8200ms", "10200ms", "10200ms"],
                "5 个节流 partial + finish 的 End"
            );
            assert_eq!(kinds(&sink.events), ["P", "P", "P", "P", "P", "E"]);
            assert_eq!(calls.lock().unwrap().len(), 6);
        }

        /// 空文本抑制:转写结果为空(或失败)时一个事件都不发。
        #[test]
        fn empty_transcriptions_emit_nothing() {
            let mut session = pseudo(
                RotatePolicy::Never,
                1_000,
                Box::new(|_| Some(text_transcript("   ".into()))),
            );
            let mut sink = Sink {
                events: Vec::new(),
                alive: true,
            };
            ambient_start(&mut session, &mut sink);
            for _ in 0..4 {
                session.push(&speech_ms(1_000), &mut sink);
            }
            session.finish(&mut sink);
            assert!(
                sink.events.is_empty(),
                "空文本不发事件: {:?}",
                kinds(&sink.events)
            );
        }

        /// 硬上限轮换:Punct 无标点可命中,`hard_ms` 强制切段——段内
        /// 先 Partial*,切段 Commit,新段重新起算,finish 出最后一段。
        #[test]
        fn hard_limit_rotates_and_commits() {
            let mut session = pseudo(
                RotatePolicy::Punct {
                    soft_ms: 60_000,
                    hard_ms: 5_000,
                },
                2_000,
                len_transcriber(Arc::new(std::sync::Mutex::new(Vec::new()))),
            );
            let mut sink = Sink {
                events: Vec::new(),
                alive: true,
            };
            ambient_start(&mut session, &mut sink);
            for _ in 0..6 {
                session.push(&speech_ms(1_000), &mut sink);
            }
            session.finish(&mut sink);
            assert_eq!(kinds(&sink.events), ["P", "P", "C", "E"]);
            assert_eq!(
                contents(&sink.events),
                ["2200ms", "4200ms", "5200ms", "1000ms"],
                "5.2s(200ms 环境静音 + 5s 语音)达硬上限切段;新段 1s 进 End"
            );
        }

        /// 标点轮换:partial 重转写文本里的新增句末标点近似 punct_ms,
        /// 到 soft 上限后的下一条 push 即切段。
        #[test]
        fn punct_rotation_commits_after_soft_limit() {
            let mut session = pseudo(
                RotatePolicy::Punct {
                    soft_ms: 2_000,
                    hard_ms: 60_000,
                },
                2_000,
                Box::new(|_| Some(text_transcript("Hello world.".into()))),
            );
            let mut sink = Sink {
                events: Vec::new(),
                alive: true,
            };
            ambient_start(&mut session, &mut sink);
            for _ in 0..3 {
                session.push(&speech_ms(1_000), &mut sink);
            }
            session.finish(&mut sink);
            // t=2.2s partial 带来标点;t=3.2s 轮换 Commit;之后段空,finish 无 End
            assert_eq!(contents(&sink.events), ["Hello world.", "Hello world."]);
            assert_eq!(kinds(&sink.events), ["P", "C"]);
        }

        /// Auto 静音门:soft 后连续 ≥500ms 静音在停顿处切段。
        #[test]
        fn auto_silence_gate_rotates_on_pause() {
            let mut session = pseudo(
                RotatePolicy::Auto {
                    soft_ms: 2_000,
                    hard_ms: 60_000,
                },
                2_000,
                len_transcriber(Arc::new(std::sync::Mutex::new(Vec::new()))),
            );
            let mut sink = Sink {
                events: Vec::new(),
                alive: true,
            };
            ambient_start(&mut session, &mut sink);
            for _ in 0..3 {
                session.push(&speech_ms(1_000), &mut sink);
            }
            for _ in 0..5 {
                session.push(&silence_ms(100), &mut sink); // 500ms 停顿
            }
            session.push(&speech_ms(1_000), &mut sink); // 停顿后的新段
            session.finish(&mut sink);
            assert_eq!(kinds(&sink.events), ["P", "C", "E"]);
            assert_eq!(
                contents(&sink.events),
                ["2200ms", "3700ms", "1000ms"],
                "t=2.2s partial;3s 语音 + 500ms 静音在停顿处切段;新段 1s 进 End"
            );
        }

        /// 全静音段跳过重转写(省 GPU):没有任何 partial 期间转写调用,
        /// finish 仍对缓冲做一次权威转写。
        #[test]
        fn all_silence_skips_retranscription() {
            let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut session = pseudo(RotatePolicy::Never, 2_000, len_transcriber(calls.clone()));
            let mut sink = Sink {
                events: Vec::new(),
                alive: true,
            };
            for _ in 0..10 {
                session.push(&silence_ms(1_000), &mut sink); // 10s 纯静音
            }
            assert!(calls.lock().unwrap().is_empty(), "全静音不重转写");
            session.finish(&mut sink);
            assert_eq!(kinds(&sink.events), ["E"]);
            assert_eq!(calls.lock().unwrap().len(), 1, "finish 一次权威转写");
        }

        /// 消费端关闭即停:首个 emit 返回 false 后不再转写,finish 无 End。
        #[test]
        fn a_closed_consumer_stops_everything() {
            let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut session = pseudo(RotatePolicy::Never, 1_000, len_transcriber(calls.clone()));
            let mut dead = Sink {
                events: Vec::new(),
                alive: false,
            };
            ambient_start(&mut session, &mut dead);
            session.push(&speech_ms(1_000), &mut dead); // since=1.2s ≥1s → partial → emit false → 关闭
            for _ in 0..3 {
                session.push(&speech_ms(1_000), &mut dead);
            }
            session.finish(&mut dead);
            assert_eq!(kinds(&dead.events), ["P"], "只有触发关闭的那一个事件");
            assert_eq!(calls.lock().unwrap().len(), 1, "关闭后不再转写");
        }
    }
}
