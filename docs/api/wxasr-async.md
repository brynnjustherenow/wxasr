# wxasr-async — 统一异步识别层

**应用的推荐入口**。专用识别线程 + tokio mpsc 事件桥;自身不做识别,
识别由 `AsrBackend` 提供(内置两个:`backend_wxasr`、`backend_gguf`)。
错误处理用 eyre(0.6)。

## 两个启动函数(`src/lib.rs`)

```rust
/// wxasr 模型专用(等价于 start_with(WxasrBackend::new(model), ..))
pub fn start(model: Arc<wxasr::Model>, cfg: Config)
    -> (AsrHandle, tokio::sync::mpsc::Receiver<AsrEvent>);

/// 通用入口:任意 AsrBackend(如 GgufBackend),audio 声明喂入格式
pub fn start_with(
    backend: Box<dyn AsrBackend>,
    cfg: Config,
    audio: AudioIn,
) -> (AsrHandle, tokio::sync::mpsc::Receiver<AsrEvent>);
```

每次调用 spawn 一个专用识别 std 线程(`Config::thread_name` 命名);
事件经有界通道(`event_capacity`)送达,识别线程内 `blocking_send`
背压——**消费端不收也会堵识别**,要尽早 `recv`。

## AsrHandle — 喂入端(Clone)

```rust
impl AsrHandle {
    pub fn push(&self, pcm: &[i16]); // 任意块大小;按 AudioIn 格式(交错);同步非阻塞
    pub fn finish(self);             // 消耗 self:定稿最后一段(空文本不发 End)+ 关闭通道
}
```

`push` 线程安全,可直接放进采集回调。

## AudioIn

```rust
pub struct AudioIn { pub sample_rate: u32, pub channels: u16 }
impl AudioIn { pub const MONO_16K: AudioIn; }  // 16_000 / 1,与 provider 输出直通
```

统一层不写死采样率:降混/重采样在各后端内部完成
(`preprocess::InputConverter`,16k mono 直通零转换)。

## Config

```rust
pub struct Config {
    pub rotate: RotatePolicy,        // 轮换(切段)策略,两后端都解释
    pub partial: bool,               // 是否产生 Partial
    pub partial_min_fresh: usize,    // partial 最小新鲜样本数(仅 wxasr 后端解释)
    pub event_capacity: usize,       // 事件通道容量,默认 64
    pub thread_name: Option<String>, // 识别线程名;None 自动 "wxasr-<uuidv7>"
}
```

## RotatePolicy(`src/policy.rs`;**文件冻结,零改动**)

```rust
pub enum RotatePolicy {
    /// soft 后遇 ≥500ms 静音轮换(静音门是冻结常量),hard 强制
    Auto { soft_ms: u32, hard_ms: u32 },
    /// soft 后遇定稿句末标点轮换;**默认 = Punct { 10000, 20000 }**
    Punct { soft_ms: u32, hard_ms: u32 },
    /// 不轮换(gguf 后端:一次会话恰一个 End)
    Never,
    Custom(Arc<dyn Fn(&RotateCtx) -> bool + Send + Sync>),
}
pub struct RotateCtx { pub stream_ms: u32, pub silence_ms: u32, pub punct_ms: Option<u32> }
```

- wxasr 后端:Punct 按**位置**切(切点干净);`rotate_at_punct`。
- gguf 后端:只有文本没有位置,`punct_ms` 取判定时刻(滞后 ≤~1s),
  会把下一句开头带进上一段——**gguf 侧更常用 Custom + 静音门**(见
  `examples/live_gguf.rs`:soft 10s + 静音 300ms、hard 30s)。
- `silence_ms` 信号:gguf 后端用 Silero VAD(缺省/失败退化能量门)。

## 事件与载荷

```rust
pub enum AsrEvent { Partial(AsrText), Commit(AsrText), End(AsrText) }

pub struct AsrText {
    pub content: String,               // 唯一必填
    pub language: Option<String>,      // ISO code
    pub speaker: Option<u32>,          // 整段同人才有;混合多人 None(分段各带)
    pub segments: Vec<AsrSegment>,
    pub words: Vec<AsrWord>,
}
pub struct AsrSegment { pub start_ms: i64, pub end_ms: i64, pub text: String, pub speaker: Option<u32> }
pub struct AsrWord { pub start_ms: i64, pub end_ms: i64, pub text: String }
```

时间戳是**会话绝对时刻**;`Partial` 不带 speaker。完整事件序与语义见
[README](README.md#统一层事件模型)。

## 自己写后端的契约(一般用不到)

```rust
pub trait AsrEvents {
    /// 返回 false = 消费端已关闭:停止后续计算、尽快收尾(再发也会被丢弃)
    fn emit(&mut self, ev: AsrEvent) -> bool;
}
pub trait AsrSession: Send {
    fn push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents);  // 识别线程内调用
    fn finish(&mut self, out: &mut dyn AsrEvents);              // 定稿并结束(空文本不发事件)
}
pub trait AsrBackend: Send + Sync {
    /// 在识别线程内调用;失败则会话立即结束,消费端看到空会话
    fn open(&self, cfg: &Config, audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>>;
}
```

## 内置后端

- `backend_wxasr::WxasrBackend::new(model: Arc<Model>)`:RMS VAD、
  partial 节流、上下文轮换(`start` 的底层)。
- `backend_gguf::GgufBackend`(feature `gguf`,编译需 CMake + Vulkan
  SDK):见 [gguf-asr.md](gguf-asr.md) 的 GgufBackend 装配一节。
  `preprocess::InputConverter::new(audio)` 供后端做格式规整。
