# wxasr

[English](README.md)

Rust 离线语音识别,构建在
[transcribe.cpp](https://github.com/brynnjustherenow/transcribe.cpp/tree/wxasr/main)
之上的高级封装(GGUF 模型:Parakeet、Whisper、Qwen3-ASR 等)。推理本身不难,
难的是应用真正需要的外围能力:不支持流式的模型也能流式出字、语音活动检测、
说话人分离、带超时与回退的专用推理线程,以及能在 Android GPU 上跑的
transcribe.cpp vendor 分支。

同时附带微信输入法 `.xnet` 模型的纯 Rust 后端,与 GGUF 后端共用同一套异步接口。

## 能力

- **统一异步接口** —— `wxasr-async` 在专用线程上识别,经 tokio 通道发事件:
  `Partial`(实时文本)、`Commit`(一段定稿,会话继续)、`End`。后端实现
  `AsrBackend` / `AsrSession` 即可接入。
- **伪流式** —— Whisper、Qwen3-ASR 没有原生流式;GGUF 后端对增长窗口重解码,
  产出与真流式模型语义一致的 partial / commit。Parakeet unified 走原生流式。
- **上下文轮换** —— 共用 `RotatePolicy`(尾随静音、句末标点、硬上限)把长会话
  切段,数小时音频下内存与延迟依然有界。
- **VAD** —— 流式 Silero VAD(GGUF)做静音判定;模型缺失或运行失败时退化为
  能量门。
- **说话人分离** —— Sortformer(≤4 人)产出说话人区间,按词级时间戳对转写
  整段切分与混合段重切。
- **稳健的引擎** —— 单推理线程、有序命令队列、前台/后台优先级、流式收尾
  超时回退整段、排队超时、GPU 健康检查与 CPU 回落。
- **跨平台 GPU** —— Windows / Linux / Android 用 Vulkan,macOS / iOS 用 Metal,
  按目标自动选择。
- **音频采集** —— `wxasr-provider` 采集麦克风与桌面回环(cpal),统一输出
  16kHz 单声道 i16。

## 工作区

```
wxasr/            纯 Rust .xnet 推理(int8 GEMM,AVX2),伪流式
wxasr-async/      统一异步层:识别线程 + tokio 事件桥,RotatePolicy;
                  后端:wxasr、gguf(feature "gguf")
wxasr-provider/   音频采集:麦克风 + 桌面回环
gguf-asr/         基于 transcribe.cpp 的 GGUF 栈:引擎、流式会话、VAD、
                  说话人分离与切分、GGUF 头部探测
vendor/           transcribe.cpp 分支(submodule,分支 wxasr/main)
```

### transcribe.cpp 分支

`vendor/transcribe.cpp` 跟随上游,并加入实机需要的修复:

- **Android Vulkan 兼容** —— `vkGetBufferDeviceAddress` 改经设备 proc address
  解析,驱动不支持时关闭 buffer device address(华为麒麟 / Maleoon GPU 原先
  会崩溃)。
- **shader 子集** —— `-DGGML_VULKAN_SHADER_TYPES` / `-DGGML_VULKAN_MUL_MAT_ID`
  只编实际用到的量化类型(嵌入 SPIR-V 约 44 MB → 7 MB)。

下游 crate 可直接依赖:

```toml
[patch.crates-io]
transcribe-cpp     = { git = "https://github.com/brynnjustherenow/transcribe.cpp", branch = "wxasr/main" }
transcribe-cpp-sys = { git = "https://github.com/brynnjustherenow/transcribe.cpp", branch = "wxasr/main" }
```

## 快速开始

离线整段转写:

```rust
use gguf_asr::{Device, Engine, ModelSpec};

let engine = Engine::load(&ModelSpec::new("Qwen3-ASR-0.6B-Q8_0.gguf"), Device::Auto)?;
let transcript = engine.transcribe_clip(samples_16k_f32)?.wait()?;
println!("{}", transcript.text);
```

链式 builder + 流式(需开启 `wxasr-async` 的 `gguf` feature):

```rust
use std::sync::Arc;
use wxasr_async::backend_gguf::{DiarizeOptions, GgufBackend, PseudoOptions};
use wxasr_async::{AsrEvent, AudioIn, Config, start_with};

let engine = Arc::new(gguf_asr::Engine::load_with_diarize(
    &spec, gguf_asr::Device::Auto, Some("diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf"),
)?);

let backend = GgufBackend::new(engine)
    .with_pseudo_streaming(PseudoOptions {           // 非流式模型也出中间结果
        vad: Some(gguf_asr::SileroVad::load("silero-vad-v6.2-F32.gguf")?),
        ..Default::default()
    })
    .with_diarize(DiarizeOptions::default())         // 说话人区间
    .with_speaker_split(gguf_asr::SpeakerSplit::Auto);

let (asr, mut rx) = start_with(
    Box::new(backend),
    Config { partial: true, ..Config::default() },
    AudioIn::MONO_16K,
);

asr.push(&pcm_i16);                                  // 在采集回调里推入
while let Some(ev) = rx.recv().await {
    match ev {
        AsrEvent::Partial(t) => println!("… {}", t.text),  // 实时,可能回改
        AsrEvent::Commit(t) | AsrEvent::End(t) => {        // 定稿;各段带发言人与时间
            for s in &t.segments { println!("[{:?}] {}", s.speaker, s.text); }
        }
    }
}
```

完整 API、轮换策略与更多用法见 [`docs/api`](docs/api/README.md)。

## 模型

本库不附带、也不指向任何模型下载。已测试的 GGUF 模型:

| 模型 | 用途 |
|---|---|
| `parakeet-unified-en-0.6b-Q8_0` | 英语,原生流式(需 GPU) |
| `whisper-large-v3-turbo-Q8_0` | 多语言,伪流式 |
| `Qwen3-ASR-0.6B-Q8_0` | 多语言(中文强),伪流式 |
| `silero-vad-v6.2-F32` | VAD |
| `diar_streaming_sortformer_4spk-v2.1-Q8_0` | 说话人分离(≤4 人) |

`.xnet` 后端需要微信输入法离线语音包(`*.xnet` 与 `dict.decoder.utf8.txt`
放同一目录),请自行获取。

## 构建

GGUF 栈要编译 transcribe.cpp 的 C++ 部分,需要 **CMake**;非 Apple 平台还需要
**Vulkan SDK**(`VULKAN_SDK`、`LIB`、`INCLUDE`)。Windows 上 `justfile` 已设好:

```powershell
just check-gguf          # 含 gguf 后端的类型检查
just live-gguf qwen      # 实时转写桌面音频
```

纯 `.xnet` 路径(`wxasr`、不开 `gguf` 的 `wxasr-async`)无原生依赖。依赖模型的
测试由 `GGUFASR_MODEL`、`GGUFASR_VAD_MODEL`、`GGUFASR_DIARIZE_MODEL` 门控。

## 致谢与许可

- `gguf-asr` 的架构受 [Hex](https://github.com/anomalyco/hex) 推理栈启发,
  独立重新实现并大幅扩展。
- `.xnet` 的容器格式、特征参数与流式策略参考了
  [J3n5en/voicekey](https://github.com/J3n5en/voicekey) 中 `wtlocal` 的行为;
  该项目无开源许可,未复用其任何代码。
- 本仓库代码 MIT;transcribe.cpp 与各模型遵循各自许可。
