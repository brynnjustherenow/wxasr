//! gguf-asr 后端:GGUF 推理栈(transcribe-cpp,parakeet/whisper/Qwen 等
//! 模型家族)接入统一层。
//!
//! [`GgufBackend`] 持有一个已加载的 [`gguf_asr::Engine`](共享的专用推理
//! 线程,模型常驻,多次会话零重复加载);每次 [`open`](AsrBackend::open)
//! 按模型能力分流:能真流式 → [`GgufStream`](流式轮换会话);不能且
//! 伪流式开启 → [`PseudoSession`];否则原行为(离线单段,只出 `End`)。
//! `Config::partial` 控制是否发中间结果;`partial_min_fresh` 忽略
//! (gguf 的流自身带节流);`Config::rotate` 由本后端解释(见下)。
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
//! - `silence_ms`:尾随静音——Silero VAD 流式概率([`GgufBackend::with_vad`],
//!   缺失或运行失败时退化为能量门,行为同 backend_wxasr 的轻量 VAD),
//!   轮换**不清零**(静音描述音频本身);
//! - `punct_ms`:committed 文本新增句末标点时的 `stream_ms`。committed
//!   即流式稳定前缀,比音频滞后 ≤~1s(解码延迟)——`Commit` 因此可能
//!   含标点之后的几个滞后词。gguf 后端无 `rotate_at_punct` 等价物,切点 =
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
//! `finish` → 最后一段 `End`。见 [`GgufBackend::pseudo_streaming`] 与
//! [`PseudoSession`]。模型能真流式时自动升级真流式路径(拿 parakeet
//! 跑伪流式后端不退化);整段转写的吞吐决定 partial 节奏与 Commit
//! 后向延迟(段长/RTF,伪流式的固有代价),阻塞期间音频排队不丢。
//!
//! 富字段直通:[`AsrText::language`] 来自模型自报的语言识别
//! (whisper 等),词/段时间戳字段在引擎以时间戳级别运行时填充
//! (默认整段路径不带时间戳,字段为空,见
//! [`gguf_asr::Transcript`])。
//!
//! # 说话人分离(diarization)
//!
//! 引擎带独立 diarize 模型(`Engine::load_with_diarize`,sortformer)时,
//! 后端持 [`DiarizeTrack`]:滚动会话缓冲(轮换不清零、按 `window_ms`
//! 截尾)+ 定稿时的一次 diarize run。sortformer 的 speaker_id 只在单次
//! run 内自洽,按滚动窗口重跑使窗口内 ID 稳定(plan §4 方案 A;
//! `window_ms = 0` 退化为仅当前段,方案 B)。装配(流式与伪流式同一套):
//! **Commit/End 时** `diarize(窗口)` → turns 折算到段时间轴 →
//! `transcribe_clip_turns(段, turns, split)` 整段权威转写并归并 speaker →
//! [`asr_text`] 直通(全段同人上浮整段 `speaker`,混合多人时顶层
//! `speaker = None`、分段各带)。`Partial` 永不带
//! speaker(契约:富字段只在定稿到齐)。流式路径由此用权威转写取代
//! native 流定稿(轮换 = 关流不排空 + 整段重转);diarize 未配置/加载
//! 失败/run 失败一律降级 `speaker = None` + warn once,转写主路径与无
//! diarize 时零差异。
//!
//! **说话人切分(speaker split)**:归并之上,把"谁在哪换的话"变成分段
//! 边界的增强,由 `split`(`SpeakerSplit`,默认 `Auto` = 引擎按模型真值
//! 解析,加载前的头部快猜见 `gguf_asr::speaker_split_for`)控制:
//! - 无时间戳模型(qwen 整段只有一条 segment,无从按时间戳归并)→
//!   **整段切分转写**:按 turns 在说话人变化处切音频、逐片转写,一次
//!   定稿携带多段多 speaker(`gguf_asr::split_transcript`);
//! - 有时间戳模型(whisper/parakeet)→ **仅混合段重切**:保留模型自己的
//!   分段与时间戳,只对区间内部跨了换人点的段切分重转写
//!   (`gguf_asr::resplit_mixed_segments`),其余段零额外转写。
//! `Off` 退回纯归并。builder 上 `with_speaker_split` 以 typestate 只挂在
//! `with_diarize` 之后(split 依赖 diarize,顺序错是编译错误);引擎未
//! 装配 diarize 模型时 turns 恒空,split 无害无效(纵深防御)。

use std::marker::PhantomData;
use std::sync::Arc;

use crate::policy::{RotateCtx, RotatePolicy};
use crate::preprocess::InputConverter;
use crate::{
    AsrBackend, AsrEvent, AsrEvents, AsrSegment, AsrSession, AsrText, AsrWord, AudioIn, Config,
};
use gguf_asr::{
    Engine, ModelSpec, SessionEvent, SileroVad, SpeakerSplit, SpeakerTurn, StreamingSession,
    Transcript,
};

/// 句末标点(与 policy 的 `Punct` 语义一致:切在句界,Commit 以完整句
/// 收尾;逗号/顿号不是句界,不触发)
const SENTENCE_END: &[char] = &['.', '!', '?', '。', '！', '？'];

/// builder 状态标记(axum `Router<S>` 式 typestate):说话人切分依赖
/// diarize,`with_speaker_split` 只存在于 [`HasDiarize`] 状态的
/// [`GgufBackend`] 上——没走过 `with_diarize` 就调它是编译错误。
pub struct NoDiarize;
/// [`GgufBackend`] 的"已配置 diarize"状态:`with_speaker_split` 从这里
/// 可达。见 [`NoDiarize`]。
pub struct HasDiarize;

/// gguf-asr 后端:封装一个已加载的引擎。
///
/// 引擎的加载(含模型加载与设备选择)是重操作,由调用方完成并共享:
/// `GgufBackend::new(Arc<Engine>)`;或用 [`GgufBackend::from_spec`] 一步
/// 加载。多个并发会话共享同一引擎(引擎内部按序调度)。
///
/// 能力按 builder 链叠加(`new` 起链,链尾即成品,无需 build):
/// `GgufBackend::new(engine).with_pseudo_streaming(..).with_diarize(..)
/// .with_speaker_split(..)`——`with_speaker_split` 类型上只挂在
/// `with_diarize` 之后(见 [`NoDiarize`])。存量组合构造器
/// (`with_pseudo`/`with_pseudo_and_diarize`/…)保留原签名。
pub struct GgufBackend<S = NoDiarize> {
    engine: Arc<Engine>,
    /// 常驻的 Silero VAD 模型;每个会话 fork 独立实例(流位置与静音
    /// 累计是会话级状态,不能串扰)。`None` → 会话用能量门。VAD 只
    /// `Send` 不 `Sync`,包 Mutex 让后端满足 `Sync`(锁只在 fork 时持有)
    vad: Option<std::sync::Mutex<SileroVad>>,
    /// `Some(partial_every_ms)` = 伪流式开启:模型在此设备不能真流式
    /// 时会话走 [`PseudoSession`](能真流式则升级真流式路径,不退化)。
    /// `None` = 原行为。
    pseudo: Option<u32>,
    /// `Some(opts)` = 说话人分离开启(引擎须带 diarize 模型加载,
    /// `Engine::load_with_diarize`;未带则装配自动降级 speaker=None)。
    diarize: Option<DiarizeOptions>,
    /// 定稿的说话人切分策略,默认 `Auto`(按引擎真值:无时间戳模型整段
    /// 切分、有时间戳模型仅混合段重切)。仅 diarize 实际装配时生效。
    split: SpeakerSplit,
    /// 术语表(优先级序):每个会话开启时交给引擎,该会话的转写与流偏向
    /// 这些词。按模型能力适配、不分家族——不支持的模型忽略并告警
    /// (`Engine::accepts_vocabulary`)。空 = 不用术语表。
    vocabulary: Vec<String>,
    /// builder 状态(typestate),运行期不参与任何行为。
    _state: PhantomData<S>,
}

/// 伪流式专属参数([`GgufBackend::with_pseudo`] 的后端构造参数,不进
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

/// 说话人分离专属参数([`GgufBackend::with_pseudo_and_diarize`] 的后端
/// 构造参数,不进 `Config`——契约:后端专属配置走构造函数)。引擎须以
/// [`gguf_asr::Engine::load_with_diarize`] 带独立 diarize 模型加载;未带
/// 则装配自动降级(转写正常,`speaker = None`)。
pub struct DiarizeOptions {
    /// diarize run 的滚动会话窗口(毫秒):每次定稿对窗口内音频重跑,
    /// 使 speaker_id 在窗口内稳定(plan §4 方案 A)。默认 120_000。
    /// `0` = 仅当前段(方案 B,跨段 ID 可能翻转)。
    pub window_ms: u32,
}

impl Default for DiarizeOptions {
    fn default() -> Self {
        DiarizeOptions { window_ms: 120_000 }
    }
}

impl GgufBackend<NoDiarize> {
    /// 用已加载的引擎创建(推荐:调用方控制加载时机与设备),也是
    /// builder 链的起点。
    pub fn new(engine: Arc<Engine>) -> Self {
        GgufBackend {
            engine,
            vad: None,
            pseudo: None,
            diarize: None,
            split: SpeakerSplit::Auto,
            vocabulary: Vec::new(),
            _state: PhantomData,
        }
    }

    /// 用已加载的引擎 + 流式 Silero VAD 创建:静音信号走 DNN 概率
    /// 而非能量门(`Auto` 族轮换策略的 endpointing 质量)。
    pub fn with_vad(engine: Arc<Engine>, vad: SileroVad) -> Self {
        GgufBackend {
            engine,
            vad: Some(std::sync::Mutex::new(vad)),
            pseudo: None,
            diarize: None,
            split: SpeakerSplit::Auto,
            vocabulary: Vec::new(),
            _state: PhantomData,
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
        GgufBackend {
            engine,
            vad: opts.vad.map(std::sync::Mutex::new),
            pseudo: Some(opts.partial_every_ms),
            diarize: None,
            split: SpeakerSplit::Auto,
            vocabulary: Vec::new(),
            _state: PhantomData,
        }
    }

    /// 加载并创建。`Engine::load` 的直通(模型加载、设备选择、预热)。
    pub fn from_spec(spec: &ModelSpec, device: gguf_asr::Device) -> eyre::Result<Self> {
        Ok(GgufBackend {
            engine: Arc::new(Engine::load(spec, device)?),
            vad: None,
            pseudo: None,
            diarize: None,
            split: SpeakerSplit::Auto,
            vocabulary: Vec::new(),
            _state: PhantomData,
        })
    }

    /// [`Self::with_vad`] 与 [`Self::from_spec`] 的组合。
    pub fn from_spec_with_vad(
        spec: &ModelSpec,
        device: gguf_asr::Device,
        vad: SileroVad,
    ) -> eyre::Result<Self> {
        Ok(GgufBackend {
            engine: Arc::new(Engine::load(spec, device)?),
            vad: Some(std::sync::Mutex::new(vad)),
            pseudo: None,
            diarize: None,
            split: SpeakerSplit::Auto,
            vocabulary: Vec::new(),
            _state: PhantomData,
        })
    }
}

/// 状态保持与状态迁移的 builder 方法:任意状态可用(`with_diarize`
/// 进入 [`HasDiarize`],重复调用 = 改窗口)。
impl<S> GgufBackend<S> {
    /// 伪流式(builder 版,任意状态可叠加):离线模型也获得完整事件
    /// 行为,能真流式则自动升级。等价伪流式旧构造器的能力,选项同
    /// [`PseudoOptions`]。
    pub fn with_pseudo_streaming(mut self, opts: PseudoOptions) -> Self {
        self.vad = opts.vad.map(std::sync::Mutex::new);
        self.pseudo = Some(opts.partial_every_ms);
        self
    }

    /// 术语表(任意状态可叠加):每个会话开启时把 `terms`(优先级序,
    /// 如人名、行话)交给引擎,该会话的转写偏向这些词。按模型能力适配、
    /// 不分家族:怎么用术语由模型自己决定(有的当提示词、有的重打分),
    /// 不支持的模型忽略并告警(`Engine::accepts_vocabulary`)。空列表 =
    /// 不用术语表。
    pub fn with_vocabulary(mut self, terms: Vec<String>) -> Self {
        self.vocabulary = terms;
        self
    }

    /// 说话人分离(进入 [`HasDiarize`] 状态,`with_speaker_split` 从此
    /// 可达):定稿时对滚动窗口跑 diarize、归并 speaker。引擎须以
    /// `Engine::load_with_diarize` 带独立 diarize 模型加载;未带则装配
    /// 自动降级(转写正常,`speaker = None`)。
    pub fn with_diarize(self, opts: DiarizeOptions) -> GgufBackend<HasDiarize> {
        let GgufBackend {
            engine,
            vad,
            pseudo,
            diarize: _,
            split,
            vocabulary,
            _state,
        } = self;
        GgufBackend {
            engine,
            vad,
            pseudo,
            diarize: Some(opts),
            split,
            vocabulary,
            _state: PhantomData,
        }
    }
}

/// "已配置 diarize"状态专属方法。
impl GgufBackend<HasDiarize> {
    /// 定稿的说话人切分策略(默认 `Auto` = 引擎真值:无时间戳模型整段
    /// 切分,有时间戳模型仅跨换人点的段重切;`Off` 关闭增强,退回整段
    /// winner-takes-all)。类型上只在此状态存在——切分依赖 diarize。
    /// 运行期纵深防御:引擎未装配 diarize 模型时 turns 为空,自动走
    /// 普通转写,本设置无害无效。
    pub fn with_speaker_split(mut self, split: SpeakerSplit) -> Self {
        self.split = split;
        self
    }

    /// 伪流式 + 说话人分离([`Self::with_pseudo`] 与 diarize 的组合):
    /// Commit/End 事件带 speaker(引擎需 `load_with_diarize`)。见模块
    /// 文档"说话人分离"一节。返回 [`HasDiarize`] 状态,可继续
    /// `with_speaker_split`。
    pub fn with_pseudo_and_diarize(
        engine: Arc<Engine>,
        opts: PseudoOptions,
        diarize: DiarizeOptions,
    ) -> Self {
        GgufBackend {
            engine,
            vad: opts.vad.map(std::sync::Mutex::new),
            pseudo: Some(opts.partial_every_ms),
            diarize: Some(diarize),
            split: SpeakerSplit::Auto,
            vocabulary: Vec::new(),
            _state: PhantomData,
        }
    }
}

/// diarize run 的会话侧入口(`DiarizeTrack` 持有;真实现走引擎命令,
/// 单测注入假 runner 记录输入/脚本输出)。输入 16 kHz f32,返回相对
/// 输入起点的 turns;`None` = 运行失败(调用方降级)。
type TurnRunner = Box<dyn FnMut(&[f32]) -> Option<Vec<SpeakerTurn>> + Send>;

/// 说话人分离的会话侧状态(流式/伪流式共用):滚动会话缓冲 + diarize
/// run 入口。轮换**不清零**缓冲(静音/语音描述音频本身),按 `window_ms`
/// 截尾;段音频由各会话自持。见模块文档"说话人分离"一节。
struct DiarizeTrack {
    run: TurnRunner,
    rolling: Vec<f32>,
    /// 滚动窗口毫秒;0 = 仅当前段(plan §4 方案 B)
    window_ms: u32,
    /// run 失败已告警(只 warn 一次,后续静默降级)
    warned: bool,
}

impl DiarizeTrack {
    /// 真实现:引擎 `diarize` 命令(专用推理线程按序执行)。
    fn new(engine: &Arc<Engine>, window_ms: u32) -> Self {
        let engine = engine.clone();
        DiarizeTrack {
            run: Box::new(move |pcm| engine.diarize(pcm).ok()?.wait().ok()),
            rolling: Vec::new(),
            window_ms,
            warned: false,
        }
    }

    /// 单测注入口:假 runner 驱动窗口/折算逻辑。
    #[cfg(test)]
    fn with_runner(run: TurnRunner, window_ms: u32) -> Self {
        DiarizeTrack {
            run,
            rolling: Vec::new(),
            window_ms,
            warned: false,
        }
    }

    /// 喂一块 16 kHz f32:入滚动缓冲并按窗口截尾。
    fn push(&mut self, pcm: &[f32]) {
        self.rolling.extend_from_slice(pcm);
        let window = self.window_ms as usize * 16;
        if self.rolling.len() > window {
            let excess = self.rolling.len() - window;
            self.rolling.drain(..excess);
        }
    }

    /// 定稿装配第一步:对窗口(`window_ms = 0` 时为段本身)跑 diarize,
    /// 返回**相对当前段起点**的 turns(与段转写的时间轴对齐;窗口与段
    /// 同终点,平移量 = 段长 − 窗口长,同终点相消无需会话时钟)。
    ///
    /// - `Some(turns)`:正常(空 turns = 全静音段跳过 diarize);
    /// - `None`:run 失败(warn once),调用方降级为无 speaker 转写。
    fn turns_for_segment(
        &mut self,
        segment: &[f32],
        speech_seen: bool,
    ) -> Option<Vec<SpeakerTurn>> {
        if !speech_seen {
            return Some(Vec::new());
        }
        let pcm: &[f32] = if self.window_ms == 0 {
            segment
        } else {
            &self.rolling
        };
        if pcm.is_empty() {
            return Some(Vec::new());
        }
        let turns = match (self.run)(pcm) {
            Some(turns) => turns,
            None => {
                if !self.warned {
                    self.warned = true;
                    tracing::warn!("diarize run 失败,本次及后续定稿不带说话人(降级,转写不受影响)");
                }
                return None;
            }
        };
        let shift = (segment.len() as i64 - pcm.len() as i64) / 16;
        Some(
            turns
                .into_iter()
                .map(|mut turn| {
                    turn.start_ms += shift;
                    turn.end_ms += shift;
                    turn
                })
                .collect(),
        )
    }
}

/// Commit/End 的段定稿转写(流式/伪流式共用):有 diarize 时先取 turns
/// 再做带归并的权威转写,失败/未配置降级为普通整段转写。
fn transcribe_segment(
    transcribe: &mut SegmentTranscriber,
    diarize: &mut Option<DiarizeTrack>,
    segment: &[f32],
    speech_seen: bool,
) -> Option<Transcript> {
    let turns = diarize
        .as_mut()
        .and_then(|track| track.turns_for_segment(segment, speech_seen));
    transcribe(segment, turns.as_deref())
}

impl<S: Send + Sync> AsrBackend for GgufBackend<S> {
    fn open(&self, cfg: &Config, audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>> {
        // 术语表先于本会话的流/转写命令入队(同一命令通道,按序生效);
        // 空列表同样下发,清掉上一个会话留在引擎里的术语。
        self.engine.set_vocabulary(self.vocabulary.clone());
        // gguf-asr 的会话吃 16 kHz 单声道 i16:调用方声明的格式在这里转换,
        // 16 kHz 单声道直通。
        let convert = InputConverter::new(audio)
            .map_err(|reason| eyre::eyre!("gguf-asr 不支持该音频格式: {reason}"))?;
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
        // diarize 后端开启但引擎未带 diarize 模型:构造 track 也无妨——
        // 首次 run 即失败降级(warn once),转写主路径零影响
        let diarize = self
            .diarize
            .as_ref()
            .map(|opts| DiarizeTrack::new(&self.engine, opts.window_ms));
        // 伪流式仅在模型此设备不能真流式时生效;能流式则自动升级
        // (伪流式后端拿 parakeet 也不退化)
        if let Some(partial_every_ms) = self.pseudo
            && !self.engine.streams()
        {
            return Ok(Box::new(PseudoSession::new(
                self.engine.clone(),
                convert,
                silence,
                diarize,
                cfg.rotate.clone(),
                cfg.partial,
                partial_every_ms,
                self.split,
            )));
        }
        Ok(Box::new(GgufStream {
            engine: self.engine.clone(),
            session: StreamingSession::open(self.engine.clone(), cfg.partial),
            // diarize 定稿装配的整段转写入口(仅 diarize 开启时使用)
            transcribe: engine_segment_transcriber(self.engine.clone(), self.split),
            diarize,
            convert,
            partial: cfg.partial,
            rotate: cfg.rotate.clone(),
            silence,
            f32: Vec::new(),
            segment: Vec::new(),
            speech_seen: false,
            stream_ms: 0,
            session_ms: 0,
            punct_ms: None,
            committed_len: 0,
        }))
    }
}

/// 真实现的"转写一段"(引擎整段转写/带归并转写):与
/// [`PseudoSession::new`] 共用,`GgufStream` 的 diarize 定稿也走它。
/// `split` 透传引擎决定定稿的说话人切分策略(`Auto` 由引擎按模型
/// 真值解析)。
fn engine_segment_transcriber(engine: Arc<Engine>, split: SpeakerSplit) -> SegmentTranscriber {
    Box::new(move |segment, turns| {
        let pending = match turns {
            Some(turns) if !turns.is_empty() => {
                engine.transcribe_clip_turns(segment.to_vec(), turns.to_vec(), split)
            }
            _ => engine.transcribe_clip(segment.to_vec()),
        };
        pending
            .ok()?
            .wait()
            .ok()
            .map(|transcribed| transcribed.transcript)
    })
}

/// gguf-asr 会话:一个 [`StreamingSession`] + 输入格式转换 + 轮换判定。
pub struct GgufStream {
    /// 轮换时开新流用(共享引擎,内部按序调度)
    engine: Arc<Engine>,
    session: StreamingSession,
    /// diarize 定稿装配的整段转写入口(仅 diarize 开启时使用)
    transcribe: SegmentTranscriber,
    /// 说话人分离状态(`None` = 未配置,定稿走 native 流文本)
    diarize: Option<DiarizeTrack>,
    convert: InputConverter,
    /// 开新流时透传给 `StreamingSession::open`
    partial: bool,
    rotate: RotatePolicy,
    /// 静音信号源(VAD 或能量门),轮换不清零
    silence: SilenceSource,
    /// i16 → f32 转换 scratch(喂 VAD/diarize 缓冲用)
    f32: Vec<f32>,
    /// 当前段音频(16 kHz f32,轮换清零;仅 diarize 开启时累积——
    /// 定稿的权威转写输入,取代 native 流定稿)
    segment: Vec<f32>,
    /// 当前段是否出现过语音(全静音段跳过 diarize)
    speech_seen: bool,
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

impl GgufStream {
    /// 喂一块 16 kHz 单声道 i16:静音信号与识别会话并行推进,更新计时
    /// 与标点游标,再按策略判定轮换。
    fn drive(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
        let ms = pcm.len() as u32 / 16; // 16000 Hz → 16 样本/ms

        // VAD 吃 f32;会话内部自己做 i16→f32。先后无所谓(互不依赖)
        self.f32.clear();
        gguf_asr::audio::i16_to_f32(pcm, &mut self.f32);
        let (silence_ms, is_speech) = self.silence.feed(pcm, &self.f32);
        self.speech_seen |= is_speech;
        if self.diarize.is_some() {
            self.segment.extend_from_slice(&self.f32);
            if let Some(track) = &mut self.diarize {
                track.push(&self.f32);
            }
        }
        self.session.push(
            pcm,
            &mut Bridge {
                out,
                final_as: FinalAs::End,
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

    /// 段状态归零并开新流(VAD 静音不清零:静音描述音频本身,
    /// policy 语义,同 backend_wxasr)。
    fn start_new_stream(&mut self) {
        self.session = StreamingSession::open(self.engine.clone(), self.partial);
        self.stream_ms = 0;
        self.punct_ms = None;
        self.committed_len = 0;
        self.speech_seen = false;
        self.segment.clear();
    }

    /// diarize 定稿装配:native 流**不排空直接关**(文本弃用,段音频
    /// 已自持;关流的 Cancel 在命令队列先于后续转写 job,引擎按序释放
    /// compute 租约)→ 段权威转写(带 turns 归并)。失败降级为无
    /// speaker 转写;转写失败返回 `None`(调用方按现状告警)。
    fn commit_segment_with_turns(&mut self) -> Option<Transcript> {
        self.session.close();
        let segment = std::mem::take(&mut self.segment);
        transcribe_segment(
            &mut self.transcribe,
            &mut self.diarize,
            &segment,
            self.speech_seen,
        )
    }

    /// 定稿当前段([`AsrEvent::Commit`])并开新流:native 流一次性,
    /// 轮换 = 老流终结 + 新流从零。无 diarize 时老流 finish 排空滞后
    /// 尾巴(文本即 Commit);有 diarize 时关流不排空,整段权威转写
    /// (带 speaker 归并)取代流定稿——流式/伪流式同一装配。
    fn rotate_stream(&mut self, out: &mut dyn AsrEvents) {
        let offset = self.stream_offset_ms();
        if self.diarize.is_some() {
            let transcript = self.commit_segment_with_turns();
            self.start_new_stream();
            emit_final(transcript, out, FinalAs::Commit, offset);
            return;
        }
        self.session.finish(&mut Bridge {
            out,
            final_as: FinalAs::Commit,
            offset,
        });
        self.start_new_stream();
    }
}

impl AsrSession for GgufStream {
    fn push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents) {
        let converted = self.convert.push(pcm);
        if converted.is_empty() {
            return;
        }
        self.drive(&converted, out);
    }

    fn finish(&mut self, out: &mut dyn AsrEvents) {
        // 重采样残留并入段缓冲(不再参与轮换判定),再定稿最后一段
        let tail = self.convert.flush();
        if !tail.is_empty() {
            self.f32.clear();
            gguf_asr::audio::i16_to_f32(&tail, &mut self.f32);
            if self.diarize.is_some() {
                self.segment.extend_from_slice(&self.f32);
                if let Some(track) = &mut self.diarize {
                    track.push(&self.f32);
                }
            } else {
                self.session.push(
                    &tail,
                    &mut Bridge {
                        out,
                        final_as: FinalAs::End,
                        offset: 0,
                    },
                );
            }
        }
        let offset = self.stream_offset_ms();
        if self.diarize.is_some() {
            let transcript = self.commit_segment_with_turns();
            emit_final(transcript, out, FinalAs::End, offset);
            return;
        }
        self.session.finish(&mut Bridge {
            out,
            final_as: FinalAs::End,
            offset,
        }); // → End
    }
}

/// [`PseudoSession`] 的"转写一段":输入当前段缓冲与可选的 diarize
/// turns(已折算到段时间轴;`Some` 且非空 → 走带归并的权威转写),
/// 返回整段转写结果(`None` = 转写失败,放弃本段文本)。真实现走引擎
/// 整段转写;单测注入假转写器(按段内容/长度生成文本、断言 turns)。
type SegmentTranscriber =
    Box<dyn FnMut(&[f32], Option<&[SpeakerTurn]>) -> Option<Transcript> + Send>;

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
    /// 说话人分离状态(`None` = 未配置,定稿不带 speaker)
    diarize: Option<DiarizeTrack>,
    convert: InputConverter,
    /// 当前段缓冲(16 kHz 单声道 f32,轮换清零)
    segment: Vec<f32>,
    /// 静音信号源(VAD 或能量门),轮换不清零
    silence: SilenceSource,
    /// 本段是否出现过语音(全静音不重转写、不 diarize)
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
    /// 真实现的"转写一段":引擎整段转写/带 turns 归并(dictation 级
    /// 优先级,引擎内部按序调度;含模型的 padding/trailing silence)。
    /// `split` 透传引擎(说话人切分策略)。
    fn new(
        engine: Arc<Engine>,
        convert: InputConverter,
        silence: SilenceSource,
        diarize: Option<DiarizeTrack>,
        rotate: RotatePolicy,
        partial: bool,
        partial_every_ms: u32,
        split: SpeakerSplit,
    ) -> Self {
        Self::with_transcriber(
            engine_segment_transcriber(engine, split),
            convert,
            silence,
            diarize,
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
        diarize: Option<DiarizeTrack>,
        rotate: RotatePolicy,
        partial: bool,
        partial_every_ms: u32,
    ) -> Self {
        PseudoSession {
            transcribe,
            diarize,
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
        gguf_asr::audio::i16_to_f32(pcm, &mut self.f32);
        let (silence_ms, is_speech) = self.silence.feed(pcm, &self.f32);
        self.speech_seen |= is_speech;
        self.stream_ms = self.stream_ms.saturating_add(ms);
        self.session_ms = self.session_ms.saturating_add(ms);
        self.since_transcribe_ms = self.since_transcribe_ms.saturating_add(ms);
        self.segment.extend_from_slice(&self.f32);
        if let Some(track) = &mut self.diarize {
            track.push(&self.f32);
        }

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
        match (self.transcribe)(&self.segment, None) {
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

    /// 段权威转写(带 diarize turns 归并)→ [`AsrEvent::Commit`]
    /// (空文本不发;消费端关闭即停)。时间戳平移到会话时间轴
    /// (契约:相对会话起点)。
    fn commit_segment(&mut self, out: &mut dyn AsrEvents) {
        if self.segment.is_empty() {
            return;
        }
        let offset = self.segment_offset_ms();
        let transcript = transcribe_segment(
            &mut self.transcribe,
            &mut self.diarize,
            &self.segment,
            self.speech_seen,
        );
        match transcript {
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
            gguf_asr::audio::i16_to_f32(&tail, &mut self.f32);
            self.segment.extend_from_slice(&self.f32);
            if let Some(track) = &mut self.diarize {
                track.push(&self.f32);
            }
        }
        if self.segment.is_empty() {
            return;
        }
        let offset = self.segment_offset_ms();
        let transcript = transcribe_segment(
            &mut self.transcribe,
            &mut self.diarize,
            &self.segment,
            self.speech_seen,
        );
        match transcript {
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
                    // Carry the silence run the VAD had built: restarting it
                    // at this chunk would delay a silence rotation already due.
                    let quiet_ms = if speech {
                        0
                    } else {
                        vad.silence_ms().saturating_add(ms)
                    };
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

/// [`SessionEvent::Final`] 映射到哪个统一层事件:轮换定稿 → `Commit`,
/// 会话结束 → `End`。
#[derive(Clone, Copy)]
enum FinalAs {
    Commit,
    End,
}

/// gguf-asr 的 [`SessionEvent`] → 统一层的 [`AsrEvent`]。`Final` 只在
/// session finish 时出现。时间戳按 `offset` 平移到会话时间轴(转写
/// 结果相对本次流起点)。
struct Bridge<'a> {
    out: &'a mut dyn AsrEvents,
    /// 本次 finish 的定稿映射
    final_as: FinalAs,
    /// 本次流起点的会话偏移(毫秒);Partial 无时间戳,忽略
    offset: i64,
}

impl gguf_asr::SessionEvents for Bridge<'_> {
    fn emit(&mut self, event: SessionEvent) -> bool {
        match event {
            SessionEvent::Partial(text) => self.out.emit(AsrEvent::Partial(AsrText::plain(text))),
            SessionEvent::Final(transcript) => {
                emit_final(Some(transcript), self.out, self.final_as, self.offset);
                true
            }
        }
    }
}

/// 一份定稿转写 → 统一层定稿事件(空文本不发;时间戳平移到会话
/// 时间轴)。diarize 装配(流式/伪流式)与 Bridge 的流定稿共用。
fn emit_final(
    transcript: Option<Transcript>,
    out: &mut dyn AsrEvents,
    final_as: FinalAs,
    offset_ms: i64,
) {
    let Some(mut transcript) = transcript else {
        return;
    };
    offset_transcript(&mut transcript, offset_ms);
    if transcript.text.trim().is_empty() {
        return;
    }
    let text = asr_text(transcript);
    match final_as {
        FinalAs::Commit => {
            out.emit(AsrEvent::Commit(text));
        }
        FinalAs::End => {
            out.emit(AsrEvent::End(text));
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

/// [`gguf_asr::Transcript`] → [`AsrText`]:富字段直通,时间戳毫秒原样。
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
            gguf_asr::audio::i16_to_f32(pcm, &mut out);
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
                gguf_asr::TranscriptSegment {
                    start_ms: 0,
                    end_ms: 500,
                    text: "hello".into(),
                    speaker: Some(1),
                },
                gguf_asr::TranscriptSegment {
                    start_ms: 500,
                    end_ms: 900,
                    text: "world".into(),
                    speaker: Some(1),
                },
            ],
            words: vec![gguf_asr::TranscriptSegment {
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
                gguf_asr::TranscriptSegment {
                    start_ms: 0,
                    end_ms: 100,
                    text: "a".into(),
                    speaker: Some(1),
                },
                gguf_asr::TranscriptSegment {
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
        pub(super) fn speech_ms(ms: u32) -> Vec<i16> {
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
            let transcribe: SegmentTranscriber = Box::new(|segment: &[f32], _| {
                let ms = segment.len() as i64 / 16;
                Some(Transcript {
                    text: format!("{ms}ms"),
                    language: None,
                    segments: vec![gguf_asr::TranscriptSegment {
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
                None,
                rotate,
                true,
                partial_every_ms,
            )
        }

        /// 按段长度生成文本的假转写器,记录每次调用的段长(样本数)
        fn len_transcriber(calls: Arc<std::sync::Mutex<Vec<usize>>>) -> SegmentTranscriber {
            Box::new(move |segment: &[f32], _| {
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
                Box::new(|_, _| Some(text_transcript("   ".into()))),
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
                Box::new(|_, _| Some(text_transcript("Hello world.".into()))),
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

    // ── DiarizeTrack:窗口截尾、折算、降级 ──────────────────

    mod diarize {
        use super::super::*;
        use super::pseudo::speech_ms;

        fn turn(start_ms: i64, end_ms: i64, speaker: u32) -> SpeakerTurn {
            SpeakerTurn {
                start_ms,
                end_ms,
                speaker,
                p: 1.0,
            }
        }

        /// 记录每次 run 输入长度的假 runner,永远返回给定 turns
        fn recording_runner(
            calls: Arc<std::sync::Mutex<Vec<usize>>>,
            turns: Vec<SpeakerTurn>,
        ) -> TurnRunner {
            Box::new(move |pcm: &[f32]| {
                calls.lock().unwrap().push(pcm.len());
                Some(turns.clone())
            })
        }

        /// 窗口截尾:滚动缓冲按 window_ms 截尾,turns 折算到段时间轴
        /// (窗口与段同终点,平移量 = 段长 − 窗口长)。
        #[test]
        fn a_windowed_run_shifts_turns_onto_the_segment_timeline() {
            let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut track = DiarizeTrack::with_runner(
                recording_runner(calls.clone(), vec![turn(0, 2_000, 2)]),
                2_000,
            );
            for _ in 0..5 {
                track.push(&vec![0.1; 16_000]); // 5s 入窗
            }
            let segment = vec![0.1; 16_000]; // 当前段 = 最后 1s
            let turns = track
                .turns_for_segment(&segment, true)
                .expect("run 成功有 turns");
            assert_eq!(calls.lock().unwrap().as_slice(), [32_000], "窗口截到 2s");
            assert_eq!(
                turns.as_slice(),
                [SpeakerTurn {
                    start_ms: -1_000,
                    end_ms: 1_000,
                    speaker: 2,
                    p: 1.0,
                }],
                "段 1s、窗 2s → 平移 −1000ms"
            );
        }

        /// window_ms = 0(方案 B):diarize 输入就是段本身,零平移。
        #[test]
        fn a_zero_window_diarizes_the_segment_itself() {
            let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut track = DiarizeTrack::with_runner(recording_runner(calls.clone(), vec![]), 0);
            track.push(&vec![0.1; 48_000]); // 滚动缓冲照收(不截尾,但不用)
            let segment = vec![0.1; 16_000];
            let turns = track.turns_for_segment(&segment, true).unwrap();
            assert_eq!(
                calls.lock().unwrap().as_slice(),
                [16_000],
                "diarize 输入 = 段本身"
            );
            assert!(turns.is_empty());
        }

        /// 全静音段跳过 diarize(转写必空,省一次 run)。
        #[test]
        fn a_silent_segment_skips_the_run() {
            let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
            let mut track =
                DiarizeTrack::with_runner(recording_runner(calls.clone(), vec![]), 120_000);
            track.push(&vec![0.0; 16_000]);
            let turns = track.turns_for_segment(&vec![0.0; 16_000], false).unwrap();
            assert!(turns.is_empty());
            assert!(calls.lock().unwrap().is_empty(), "静音段不跑 diarize");
        }

        /// run 失败 → None(降级,转写不带 speaker),后续照常重试。
        #[test]
        fn a_failing_run_degrades_to_no_turns() {
            let mut track: DiarizeTrack = DiarizeTrack::with_runner(Box::new(|_| None), 120_000);
            track.push(&vec![0.1; 16_000]);
            assert!(track.turns_for_segment(&vec![0.1; 16_000], true).is_none());
            assert!(track.turns_for_segment(&vec![0.1; 16_000], true).is_none());
        }

        /// 端到端:Commit/End 的转写收到折算后的 turns(伪流式装配;
        /// 流式路径共用 [`transcribe_segment`],同一装配)。
        #[test]
        fn commits_carry_the_shifted_turns_into_the_transcription() {
            // 窗口覆盖全会话 → 平移 0;turn 覆盖 0..2s、说话人 2
            let track = DiarizeTrack::with_runner(
                recording_runner(
                    Arc::new(std::sync::Mutex::new(Vec::new())),
                    vec![turn(0, 2_000, 2)],
                ),
                120_000,
            );
            let transcribe: SegmentTranscriber = Box::new(|segment: &[f32], turns| {
                let ms = segment.len() as i64 / 16;
                Some(Transcript {
                    text: format!("{ms}ms"),
                    language: None,
                    segments: vec![gguf_asr::TranscriptSegment {
                        start_ms: 0,
                        end_ms: ms,
                        text: String::new(),
                        speaker: turns.and_then(|turns| turns.first()).map(|t| t.speaker),
                    }],
                    words: Vec::new(),
                })
            });
            let rotate = RotatePolicy::Custom(Arc::new(|ctx: &RotateCtx| ctx.stream_ms >= 2_000));
            let mut session = PseudoSession::with_transcriber(
                transcribe,
                InputConverter::new(AudioIn::MONO_16K).expect("16k mono 直通"),
                SilenceSource::default(),
                Some(track),
                rotate,
                false,
                u32::MAX,
            );
            struct Sink(Vec<AsrEvent>);
            impl AsrEvents for Sink {
                fn emit(&mut self, event: AsrEvent) -> bool {
                    self.0.push(event);
                    true
                }
            }
            let mut sink = Sink(Vec::new());
            // 能量门要环境静音播种噪声底(§8 冷启动坑),否则语音全判静音
            session.push(&vec![0_i16; 3_200], &mut sink); // 200ms 前导
            for _ in 0..25 {
                session.push(&speech_ms(100), &mut sink); // 2.5s 语音
            }
            session.finish(&mut sink);
            let speakers: Vec<Option<u32>> = sink
                .0
                .iter()
                .filter_map(|event| match event {
                    AsrEvent::Commit(text) | AsrEvent::End(text) => {
                        text.segments.first().map(|s| s.speaker)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(speakers, [Some(2), Some(2)], "Commit 与 End 都带说话人");
        }
    }
}
