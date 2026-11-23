# 接线配方

可复制的使用模式。权威实现:`wxasr-async/examples/live_gguf.rs`
(实时桌面听写 CLI,双端点最全接线)。

## 配方一:GGUF 实时听写(live_gguf 模式)

```rust
use std::sync::Arc;
use wxasr_async::backend_gguf::{DiarizeOptions, GgufBackend, PseudoOptions};
use wxasr_async::policy::{RotateCtx, RotatePolicy};
use wxasr_async::{AsrEvent, AudioIn, Config, start_with};
use wxasr_provider::{AudioSource, DesktopAudio, PcmSink};

// 1) 引擎(whisper 才接 speech_detector;VAD 在引擎之后加载)
let engine = Arc::new(gguf_asr::Engine::load_with_diarize(
    &spec, gguf_asr::Device::Auto, diarize_model.as_deref(),
)?);
let streams = engine.streams();                       // 进 backend 前取好
if gguf_asr::gguf_meta(&model)?.architecture == "whisper" && let Some(vad) = vad_path() {
    gguf_asr::speech_detector::set_model(&vad);
    gguf_asr::speech_detector::prewarm();
}
let backend = GgufBackend::new(engine.clone())
    .with_pseudo_streaming(PseudoOptions {
        vad: vad_path().map(|p| gguf_asr::SileroVad::load(&p)).transpose()?,
        ..Default::default(),
    })
    .with_diarize(DiarizeOptions::default())          // 需要时;之后才可 with_speaker_split
    .with_speaker_split(gguf_asr::SpeakerSplit::Auto);

// 2) 轮换:gguf 侧推荐 Custom + 静音门(不用默认 Punct:标点切点=判定时刻,
//    滞后 ≤~1s,会把下一句开头带进上一段)
let rotate = RotatePolicy::Custom(Arc::new(|ctx: &RotateCtx| {
    ctx.stream_ms >= 30_000 || (ctx.stream_ms >= 10_000 && ctx.silence_ms >= 300)
}));

// 3) 会话 + 采集
let (asr, mut rx) = start_with(
    Box::new(backend),
    Config { rotate, partial: true, ..Config::default() },
    AudioIn::MONO_16K,
);
let sink: PcmSink = { let asr = asr.clone(); Arc::new(move |pcm: &[i16]| asr.push(pcm)) };
let source = DesktopAudio::new();
let mut capture = source.start(None, sink)?;

// 4) 消费(需要 tokio runtime 在上下文里;或用 blocking_recv)
while let Some(ev) = rx.recv().await {
    match ev {
        AsrEvent::Partial(t) => { /* 投机文本,可回改 */ }
        AsrEvent::Commit(t) | AsrEvent::End(t) => { /* 定稿;t.segments 各带 speaker+时间 */ }
    }
}
// 停止:capture.stop(); asr.finish(); 等 End 与通道关闭
```

模型 spec:parakeet 必须 `ModelSpec::unified_english`(en 提示 + 400ms
补静音,`ModelSpec::new` 的自动检测会拒载它);其它模型
`ModelSpec::new`;`GGUFASR_LANG=<ISO>` 显式覆盖语言。

## 配方二:双流共享一个引擎(kukac 模式)

麦克风 + 系统回环各一路、共用一个模型:**一次** `load_with_diarize`,
`Arc<Engine>` clone 给两个 `GgufBackend`,`start_with` 两次
(`thread_name` 各自命名便于诊断)。引擎内部按序调度——两路的转写
job 排队执行,GPU 上 partial 节奏会互相拉长,属预期。

要点:
- cpal `Capture` **不是 `Send`**(WASAPI 流句柄):不能放进全局静态/
  跨线程移动——每路开一个专属线程持有它,线程内等停止命令后
  `capture.stop()` + 封盘 wav。
- `AsrHandle` 是 `Send + Sync`,`push` 可直接放采集回调;录音落盘在
  回调里 tee 给 `hound::WavWriter`(1ch/16k/i16,与 provider 输出一致)。
- 事件泵:tokio mpsc `Receiver::recv()` 是运行时无关 future,任意
  执行器可轮询(不强制 tokio runtime)。

## 配方三:离线整段转写

```rust
let engine = gguf_asr::Engine::load(&ModelSpec::new("model.gguf"), Device::Auto)?;
let transcript = engine.transcribe_clip(read_wav_f32())?.wait()?;
println!("{}", transcript.text);
```

`foreground` / `finish_with` 是带 `offline` 闭包的底层入口
(`pub type Offline = Box<dyn FnOnce(&mut Model, Vec<f32>) -> Result<Transcript, String> + Send>`,
注入整段转写方式);一般用 `transcribe_clip` / `finish`。

wxasr 侧:`wxasr::Model::load(dir)?.transcribe_wav("a.wav")`。

## 模型目录与约定

模型副本目录 `G:\env\model\asr`(`GGUF_MODELS_DIR` 可覆盖,live_gguf
默认目录):

| 文件 | 用途 |
| --- | --- |
| `parakeet-unified-en-0.6b-Q8_0.gguf`(731MB) | **唯一支持 native 流式 partial 的模型**(仅英语,需 GPU) |
| `whisper-large-v3-turbo-Q8_0.gguf`(845MB) | 伪流式;需接防幻觉 speech detector |
| `Qwen3-ASR-0.6B-Q8_0.gguf`(811MB) | 伪流式;无时间戳(说话人切分走 Pieces) |
| `silero-vad-v6.2-F32.gguf`(1.2MB) | Silero VAD(endpointing + whisper 防幻觉共用) |
| `diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf`(133MB) | Sortformer 说话人分离(≤4 说话人,英语训练) |

原始文件(用户经 Hex 应用下载,**勿动**)在
`C:\Users\Administrator\AppData\Local\hex\models\`。

## 作为依赖引用(git)

四个 crate 未发布 crates.io;同仓 monorepo,path 依赖在 git 检出内正常
解析。同一 URL + 同 rev,cargo 统一为一次检出:

```toml
[dependencies]
wxasr-provider = { git = "ssh://git@github.com/brynnjustherenow/wxasr.git", rev = "<commit>" }
wxasr-async    = { git = "…同上…", rev = "<commit>", features = ["gguf"] }
gguf-asr       = { git = "…同上…", rev = "<commit>" }
```

`gguf` feature 编译需要 CMake + Vulkan SDK(下节)。

## 构建环境(Windows;每条 cargo 前必设)

日常优先用根目录 `justfile`(`just --list`;gguf 配方自带环境)。手动
设四件套(PowerShell):

```powershell
$env:Path = "C:\Program Files\CMake\bin;$env:Path"
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.363.0"
$env:LIB = "C:\VulkanSDK\1.4.363.0\Lib;$env:LIB"
$env:INCLUDE = "C:\VulkanSDK\1.4.363.0\Include;$env:INCLUDE"
```

- 漏 `LIB` → LNK1181(找不到 vulkan-1.lib);已存进程拿不到新装的机器
  环境变量,必须显式设。
- transcribe-cpp-sys 的 C++ 首次/依赖树变动后要编几分钟:长命令
  timeout ≥600000ms,或后台跑 + 重定向日志;并发 cargo 按构建锁排队
  属正常。
- 测试 runner 无超时,挂死会占构建锁:按命令行杀进程树释放。

## 测试

- 门控环境变量(未设则 skip):`GGUFASR_MODEL`(真实模型测试)、
  `GGUFASR_VAD_MODEL`(VAD;缺省用模型目录默认副本,CPU 加载)、
  `GGUFASR_DIARIZE_MODEL`(说话人分离;设了才装配,`engine.diarizes()`
  为真)。
- `just test-model` 一律 `--test-threads=1`:并行实时会话争抢同一 GPU
  引擎,输家的流 decode 落后 → 0 partial(假失败,不是回归)。
- 预期基线:wxasr-async 无 feature 19+6 / `--features gguf` lib 38;
  gguf-asr lib 123(121 过 + 2 ignored 属存量);tests/model.rs 10;
  gguf_backend 集成 6。Vulkan 大模型加载几十秒,timeout 放大。

## 已知坑(踩过的)

- **快于实时喂音频不产生 partial**(流式 decode 与喂入同步推进)。
- **whisper 无 VAD 死等已修复**:根因是 speech_detector 提取时的
  自死锁(mutex 双锁),与 GPU/模型/参数无关;诊断探针在
  `gguf-asr/tests/native_probe.rs`(`GGUFASR_NATIVE_PROBE` /
  `GGUFASR_RAW_PROBE` / `GGUFASR_RAW_BISECT`)。
- CPU + whisper 会撞引擎 deadline(超时属设计内,伪流式测试自动 skip
  CPU);推理/听写一律跑 release。
- 两个 `--test-threads=1` 之外的并行会话会 GPU 争抢(见上)。
