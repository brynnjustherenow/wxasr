# gguf-asr — GGUF 推理栈

灵感来源于 Hex、在其架构上重新实现并大幅扩展的 GGUF/Vulkan 推理栈
(transcribe-cpp crates.io `=0.4.1` + `vulkan` feature,无 fork patch)。
调用链:engine(专用线程调度)→ local_transcriber → gguf_session →
transcribe-cpp。错误用 eyre。

统一层适配在 `wxasr-async` 的 `backend_gguf` 模块(本篇末尾)。

## Engine(`src/engine.rs`)— 共享引擎

`Send + Sync`:**包 `Arc` 跨会话/跨线程共享一个已加载模型**(引擎内部
单命令线程按序调度,多会话排队执行)。

```rust
impl Engine {
    // ── 构建 ────────────────────────────────
    pub fn load(spec: &ModelSpec, device: Device) -> Result<Self>;
    /// diarize = Some(sortformer gguf 路径) 时装配说话人分离;
    /// 加载失败仅 warn 降级(转写不带 speaker),不 fail ASR
    pub fn load_with_diarize(spec: &ModelSpec, device: Device, diarize: Option<&Path>) -> Result<Self>;
    pub fn start(model: Model) -> Result<Self>;                    // 已加载模型开引擎
    pub fn start_with(model: Model, diarizer: Option<Diarizer>) -> Result<Self>;

    // ── 流式(真流式模型:parakeet) ──────────
    pub fn begin(&self) -> Option<LiveStream>;                      // 不能流式 → None
    pub fn begin_ignoring_head(&self, head_ignored: usize) -> Option<LiveStream>;
    pub fn begin_yielding(&self) -> Option<YieldingStream>;
    pub fn streams(&self) -> bool;                                  // 运行时真值(含设备 probe);engine 进 backend 前取好
    pub fn busy(&self) -> bool;

    // ── 术语表(按模型能力适配,不分家族)──────
    pub fn accepts_vocabulary(&self) -> bool;                       // 模型是否吃自定义术语(加载后读一次,之后不变)
    pub fn set_vocabulary(&self, terms: Vec<String>);               // 优先级序;此后的整段转写与新建的流生效

    // ── 整段转写(f32 16k mono) ─────────────
    pub fn finish(&self, stream: Option<LiveStream>, clip: Vec<f32>) -> Result<PendingTranscript>;
    pub fn finish_with(&self, stream: Option<LiveStream>, clip: Vec<f32>, offline: Offline) -> Result<PendingTranscript>;
    pub fn foreground(&self, clip: Vec<f32>, offline: Offline) -> Result<PendingTranscript>;
    pub fn transcribe_clip(&self, clip: Vec<f32>) -> Result<PendingTranscript>;
    // Offline = Box<dyn FnOnce(&mut Model, Vec<f32>) -> Result<Transcript, String> + Send>
    // —— 整段转写方式注入点;一般用不带它的 finish / transcribe_clip

    // ── 说话人分离 ──────────────────────────
    pub fn diarizes(&self) -> bool;                                 // 引擎真的装配了 diarize
    pub fn speaker_split(&self) -> SpeakerSplit;                    // Auto 的运行时解析值
    pub fn diarize(&self, pcm: &[f32]) -> Result<PendingTurns>;     // turns 相对本次 pcm
    pub fn transcribe_clip_turns(&self, pcm: Vec<f32>, turns: Vec<SpeakerTurn>, split: SpeakerSplit)
        -> Result<PendingTranscript>;                               // 定稿按 SpeakerSplit 切分转写

    pub fn background<R: Send + 'static>(..) -> Result<Option<R>>;
    pub fn retire(self);                                            // 后台线程 drop(显式回收)
}
```

`PendingTranscript::wait(self) -> Result<Transcript, String>`;
`PendingTurns::wait(self) -> Result<Vec<SpeakerTurn>, String>`。

**术语表(vocabulary)**:自定义术语(人名、行话等,优先级序),加载时经
`ModelSpec::vocabulary`、运行时经 `Engine::set_vocabulary` 设置,进到底层
每次 run / stream 的 `RunOptions::vocabulary`。**按模型能力适配、不分家族**:
是否生效以 `Engine::accepts_vocabulary()` 为准(底层 `Feature::Vocabulary`),
不支持的模型忽略并告警;术语怎么用(当提示词还是重打分)由模型自己决定。
术语入库时 trim、去空;已开的流保留开始时的术语,变更在该流结束后生效;
GPU 失败换到 CPU 重载时术语保留;超出 prompt 预算的术语由底层从列表末尾
丢弃并告警。

**已知差异(是否开启由应用层取舍)**:三个目录家族都自报支持,但效果不同——
实测一份音频里没说到的名字表,Parakeet / Qwen3-ASR 的转写不变;Whisper 把
术语当提示词(放进 `<|startofprev|>` 槽的 `Glossary: …`),英文长句会丢
大部分句末标点,术语与录音开头的词相同时那几个词会被当成"已说过"而吞掉。

- `LiveStream::feed(&mut self, &[f32])` / `take_partial() -> Option<Partial>` /
  `fed() -> usize`(已喂样本)。
- `Transcript { text, language, segments: Vec<TranscriptSegment>, words }`;
  `TranscriptSegment { start_ms, end_ms, text, speaker }`。

## ModelSpec / Device(`src/local_transcriber.rs`)

```rust
pub enum Device { Auto, Gpu, Cpu }   // Default = Auto
pub struct ModelSpec {
    pub path: PathBuf,
    pub language: Option<String>,             // None = 自动检测
    pub trailing_silence_samples: usize,      // 16k 采样数,整段尾部补静音
    pub vocabulary: Vec<String>,             // 自定义术语(优先级序),见上文「术语表」
}
impl ModelSpec {
    pub fn new(path) -> Self;                          // language None,0 补静音
    pub fn unified_english(path) -> Self;              // en + 400ms(6400 样本):parakeet 专用
}
pub fn devices() -> Result<Vec<String>>;
```

parakeet(不自报语言检测)必须 `unified_english` 或显式给
`language`,否则 `ModelSpec::new` 的自动检测会被拒载。

## gguf_meta(`src/gguf_meta.rs`)— 加载前读头

```rust
pub struct GgufMeta { pub architecture: String, pub streaming: bool, pub name: Option<String> }
pub fn gguf_meta(path) -> eyre::Result<GgufMeta>;   // 只读 GGUF v3 头 KV,毫秒级,不加载权重
```

用途:模型目录筛选(挑 ASR 家族,排除 VAD/diarize 模型)、加载前展示
家族与"是否自报流式"(运行时真值仍以 `Engine::streams()` 为准)。
ASR 家族名(即 `speaker_split_for` 的无时间戳清单 + whisper/parakeet):
`whisper`、`parakeet`、`qwen3_asr`、`canary`、`canary_qwen`、
`cohere_asr`、`funasr_nano`、`granite_speech5_ctc`、`granite_speech_nar`、
`moonshine`、`moonshine_streaming`、`sensevoice`、`voxtral`、
`voxtral_realtime`。

## SileroVad(`src/vad.rs`)— 流式静音检测

```rust
SileroVad::load(path) -> Result<Self>   // CPU 加载,不需要 Vulkan
vad.fork() -> Result<Self>              // 每会话/每后端 fork 一份
vad.feed(&pcm_f32) / is_speech() / silence_ms() / reset()
```

供后端做轮换静音信号与伪流式全静音跳过;与 whisper 防幻觉 speech
detector 是**两个独立实例**。

## speech_detector(`src/speech_detector.rs`)— whisper 防幻觉

静音编字是 whisper 家族病理;进程级 detector 裁掉语音段落的静音边沿、
跳过长片段里的全静音 chunk。**读 gguf_meta 判明架构是 whisper 后才接**:

```rust
gguf_asr::speech_detector::set_model(&vad_path);  // 与 endpointing VAD 共用同一 silero 文件
gguf_asr::speech_detector::prewarm();             // 后台加载,首次转写不付等待
```

## diarize(`src/diarize.rs`)— Sortformer 说话人分离

```rust
pub struct SpeakerTurn { pub start_ms: i64, pub end_ms: i64, pub speaker: u32 /*1-based*/, pub p: f32 }

Diarizer::load(path: &Path, device: Device) -> Result<Self>
diarizer.max_speakers() -> i32
diarizer.run(&mut self, pcm: &[f32]) -> Result<Vec<SpeakerTurn>>          // turns 相对本次 pcm

pub fn assign_speakers(transcript: &mut Transcript, turns: &[SpeakerTurn]);  // 按时间戳归并
pub fn speaker_chunks(len_ms: i64, turns: &[SpeakerTurn], min_piece_ms: i64) -> Vec<SpeakerChunk>;
// 无时间戳模型:按 turns 切音频、逐片调用 transcribe 闭包,一次定稿多段多 speaker;
// 分片失败整体降级
pub fn split_transcript(
    pcm: &[f32], turns: &[SpeakerTurn],
    transcribe: impl FnMut(&[f32]) -> Result<Transcript>,
) -> Result<Transcript>;
// 有时间戳模型:保留模型自身分段,仅对区间内跨换人点的段重切重转写;返回重切段数
pub fn resplit_mixed_segments(
    pcm: &[f32], transcript: &mut Transcript, turns: &[SpeakerTurn],
    transcribe: impl FnMut(&[f32]) -> Result<Transcript>,
) -> usize;
pub fn speaker_split_for(meta: &GgufMeta) -> SpeakerSplit;                  // 加载前头部快猜
```

**SpeakerSplit**(定稿切分模式,`Engine::transcribe_clip_turns` 第三参):

| 变体 | 语义 |
| --- | --- |
| `Auto`(默认) | 引擎按 `finest_timestamps` 真值解析(= Pieces 或 MixedSegments);加载前快猜 `speaker_split_for(&GgufMeta)` |
| `Pieces` | 无时间戳模型(qwen3_asr 等):整段切分转写——按 turns 切音频、切点吸附 ±250ms 最静 100ms 窗、逐片转写;分片失败整体降级 |
| `MixedSegments` | 有时间戳模型(whisper/parakeet):保留自身分段,仅区间内跨换人点的段重切重转写;逐段降级 |
| `Off` | 纯归并,不切分 |

引擎未装配 diarize 时 turns 恒空,split 无害无效。qwen3_asr
`max_timestamp_kind = NONE`(恒单段 [0, 整段]),无从归并,只能靠切分。

## StreamingSession(`src/session.rs`)— 手动驱动流式会话

不经统一层、自己驱动时用(gguf 后端内部即它):

```rust
pub trait SessionEvents { .. }  // Partial(String) / Final(Transcript)
StreamingSession::open(engine: Arc<Engine>, partials: bool) -> Self
session.push(&mut self, pcm: &[i16], out: &mut dyn SessionEvents)  // 内部 16k 规整
session.finish(&mut self, out) / close() / pushed_samples() / committed_text() / is_streaming()
```

## GgufBackend 装配(wxasr-async feature `gguf`)

typestate 构造器(`axum Router<S>` 式);`with_speaker_split` 依赖
diarize,顺序错是编译错误:

```rust
use wxasr_async::backend_gguf::{DiarizeOptions, GgufBackend, PseudoOptions};

pub struct PseudoOptions { pub partial_every_ms: u32 /*默认 2000*/, pub vad: Option<SileroVad> }
pub struct DiarizeOptions { pub window_ms: u32 /*默认 120_000;0 = 仅当前段*/ }

let engine = Arc::new(Engine::load_with_diarize(&spec, device, diarize_path.as_deref())?);
// whisper 才接 speech_detector(见上);VAD 在引擎之后加载
let backend = GgufBackend::new(engine.clone())
    .with_pseudo_streaming(PseudoOptions { vad: Some(SileroVad::load(vad)?), ..Default::default() })
    .with_diarize(DiarizeOptions::default())       // → GgufBackend<HasDiarize>
    .with_speaker_split(SpeakerSplit::Auto);
// 术语表(任意状态可叠加):每个会话开启时把术语交给引擎,按模型能力生效
let backend = GgufBackend::new(engine.clone())
    .with_vocabulary(vec!["Kubernetes".into(), "gRPC".into()])
    .with_pseudo_streaming(PseudoOptions::default());
// 其它起点:with_vad(engine, vad) / pseudo_streaming(engine) / with_pseudo(engine, opts)
//   / from_spec(&spec, device) / from_spec_with_vad(..)
// 兼容组合构造器:with_pseudo_and_diarize(engine, pseudo, diarize) 亦在 HasDiarize 态
let (asr, mut rx) = wxasr_async::start_with(Box::new(backend), cfg, AudioIn::MONO_16K);
```

**伪流式**(离线模型的完整事件行为):`open` 见 `engine.streams()` 为
假且伪流式开启时走 `PseudoSession`——增长缓冲 + 每 `partial_every_ms`
整段重转写 → 变化文本当 `Partial`(允许整体回改);轮换切段处整段权威
转写 → `Commit`;`finish` → `End`。能真流式则自动升级(parakeet 跑不
退化)。`hard_ms` 即段缓冲上界;整段转写阻塞期间音频排队不丢;
Commit/End 后向延迟 = 段长/RTF(GPU 上 20s 段 ≈1s)。

**diarize 开启时的定稿**:Commit/End 不再用流式定稿,而是对滚动窗口
跑 diarize → turns 折算 → `transcribe_clip_turns` 整段权威转写并按
SpeakerSplit 切分,经统一层直通 `AsrText.segments`(带 speaker 与会话
绝对时间;顶层 speaker 仅全段同人)。**一次 Commit 可携带多段多
speaker**——"commit 会被按 speaker 重切"。
