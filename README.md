# wxasr

[简体中文](README.zh-CN.md)

Offline speech recognition for Rust, built as a high-level layer on top of
[transcribe.cpp](https://github.com/brynnjustherenow/transcribe.cpp/tree/wxasr/main)
(GGUF models: Parakeet, Whisper, Qwen3-ASR, …). Raw inference is not the hard
part; what this workspace adds is what an app actually needs around it:
streaming behaviour for models that can't stream, voice activity detection,
speaker diarization, a dedicated engine thread with deadlines and fallbacks,
and a vendored transcribe.cpp fork that runs on Android GPUs.

It also ships a pure-Rust backend for the WeChat IME `.xnet` model, behind the
same async interface.

## Features

- **One async interface for every backend** — `wxasr-async` runs recognition on
  a dedicated thread and bridges events over a tokio channel:
  `Partial` (live text), `Commit` (a finalized segment, session continues),
  `End`. Backends implement `AsrBackend` / `AsrSession`.
- **Pseudo-streaming** — Whisper and Qwen3-ASR have no native streaming; the
  GGUF backend re-decodes a growing window and emits partials / commits with
  the same event semantics as a true streaming model. Parakeet unified uses
  native streaming.
- **Context rotation** — long sessions are cut into segments by a shared
  `RotatePolicy` (trailing silence, sentence-final punctuation, hard length
  cap), so memory and latency stay bounded over hours of audio.
- **VAD** — streaming Silero VAD (GGUF) for silence endpointing, with an energy
  gate fallback when the model is missing or fails.
- **Speaker diarization** — Sortformer (≤ 4 speakers) produces speaker turns;
  transcripts are split and re-split by speaker using word timestamps.
- **Robust engine** — one inference thread, ordered command queue, foreground
  vs. background priority, stream-finish deadline that falls back to the whole
  clip, queue deadlines, GPU health checks with CPU recovery.
- **Cross-platform GPU** — Vulkan on Windows / Linux / Android, Metal on
  macOS / iOS (selected per target automatically).
- **Audio capture** — `wxasr-provider` captures microphone and desktop
  loopback (cpal) as 16 kHz mono i16.

## Workspace

```
wxasr/            pure-Rust .xnet inference (int8 GEMM, AVX2), pseudo-streaming
wxasr-async/      unified async layer: engine thread + tokio event bridge,
                  RotatePolicy; backends: wxasr, gguf (feature "gguf")
wxasr-provider/   audio capture: microphone + desktop loopback
gguf-asr/         GGUF stack over transcribe.cpp: engine, streaming session,
                  VAD, diarization, speaker split, GGUF header probing
vendor/           transcribe.cpp fork (submodule, branch wxasr/main)
```

### The transcribe.cpp fork

`vendor/transcribe.cpp` tracks upstream with fixes needed in the field:

- **Android Vulkan compatibility** — `vkGetBufferDeviceAddress` is resolved
  through the device proc address, and buffer device address is disabled when
  the driver lacks it (Huawei Kirin / Maleoon GPUs crashed otherwise).
- **Shader subset** — `-DGGML_VULKAN_SHADER_TYPES` / `-DGGML_VULKAN_MUL_MAT_ID`
  build only the quantization types you ship (embedded SPIR-V ~44 MB → ~7 MB).

Downstream crates can depend on it directly:

```toml
[patch.crates-io]
transcribe-cpp     = { git = "https://github.com/brynnjustherenow/transcribe.cpp", branch = "wxasr/main" }
transcribe-cpp-sys = { git = "https://github.com/brynnjustherenow/transcribe.cpp", branch = "wxasr/main" }
```

## Quick start

Offline, whole clip:

```rust
use gguf_asr::{Device, Engine, ModelSpec};

let engine = Engine::load(&ModelSpec::new("Qwen3-ASR-0.6B-Q8_0.gguf"), Device::Auto)?;
let transcript = engine.transcribe_clip(samples_16k_f32)?.wait()?;
println!("{}", transcript.text);
```

Streaming with the builder (needs `wxasr-async` feature `gguf`):

```rust
use std::sync::Arc;
use wxasr_async::backend_gguf::{DiarizeOptions, GgufBackend, PseudoOptions};
use wxasr_async::{AsrEvent, AudioIn, Config, start_with};

let engine = Arc::new(gguf_asr::Engine::load_with_diarize(
    &spec, gguf_asr::Device::Auto, Some("diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf"),
)?);

let backend = GgufBackend::new(engine)
    .with_pseudo_streaming(PseudoOptions {           // partials for non-streaming models
        vad: Some(gguf_asr::SileroVad::load("silero-vad-v6.2-F32.gguf")?),
        ..Default::default()
    })
    .with_diarize(DiarizeOptions::default())         // speaker turns
    .with_speaker_split(gguf_asr::SpeakerSplit::Auto);

let (asr, mut rx) = start_with(
    Box::new(backend),
    Config { partial: true, ..Config::default() },
    AudioIn::MONO_16K,
);

asr.push(&pcm_i16);                                  // from the capture callback
while let Some(ev) = rx.recv().await {
    match ev {
        AsrEvent::Partial(t) => println!("… {}", t.text),  // live, may change
        AsrEvent::Commit(t) | AsrEvent::End(t) => {        // final; segments carry speaker + time
            for s in &t.segments { println!("[{:?}] {}", s.speaker, s.text); }
        }
    }
}
```

See [`docs/api`](docs/api/README.md) for the full API, rotation policies and
more recipes.

## Models

No models are bundled or linked. Tested GGUF models:

| Model | Use |
|---|---|
| `parakeet-unified-en-0.6b-Q8_0` | English, native streaming (GPU) |
| `whisper-large-v3-turbo-Q8_0` | multilingual, pseudo-streaming |
| `Qwen3-ASR-0.6B-Q8_0` | multilingual (strong Chinese), pseudo-streaming |
| `silero-vad-v6.2-F32` | VAD |
| `diar_streaming_sortformer_4spk-v2.1-Q8_0` | speaker diarization (≤ 4) |

The `.xnet` backend needs the WeChat IME offline voice pack (`*.xnet` +
`dict.decoder.utf8.txt` in one directory), which you must obtain yourself.

## Building

The GGUF stack builds transcribe.cpp's C++ code, so it needs **CMake** and,
off Apple platforms, the **Vulkan SDK** (`VULKAN_SDK`, `LIB`, `INCLUDE`). The
`justfile` sets this up on Windows:

```powershell
just check-gguf          # type-check with the gguf backend
just live-gguf qwen      # live desktop-audio transcription
```

The pure `.xnet` path (`wxasr`, `wxasr-async` without `gguf`) has no native
dependencies. Model-dependent tests are gated by `GGUFASR_MODEL`,
`GGUFASR_VAD_MODEL` and `GGUFASR_DIARIZE_MODEL`.

## Acknowledgements & license

- `gguf-asr`'s architecture is inspired by [Hex](https://github.com/anomalyco/hex)'s
  inference stack and reimplemented with substantial extensions.
- The `.xnet` container format, features and streaming strategy follow the
  behaviour of `wtlocal` in [J3n5en/voicekey](https://github.com/J3n5en/voicekey);
  no code was reused (that project has no license).
- Code in this repository: MIT. transcribe.cpp and the models keep their own
  licenses.
