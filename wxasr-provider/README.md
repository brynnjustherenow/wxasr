# wxasr-provider

wxasr 家族的跨平台音频采集层：把设备原生格式（采样率/通道/样本类型）
统一转换为 **16kHz 单声道 i16 PCM** 投递给回调。

## 提供者

| 提供者 | 平台 | 机制 |
|---|---|---|
| `Microphone` | 全平台 | 常规输入设备 |
| `DesktopAudio` | Windows | WASAPI loopback（输出设备上建输入流，捕获系统混音）|
| | Linux | PulseAudio/PipeWire monitor 源（按名字含 "monitor" 自动选择）|
| | macOS | 需 BlackHole 等虚拟设备（以输入设备形式出现）|

## 用法

```rust
use std::sync::Arc;
use wxasr_provider::{AudioSource, DesktopAudio};

let src = DesktopAudio::new();
let mut capture = src.start(None, Arc::new(|pcm: &[i16]| {
    // 16kHz 单声道 i16，直接喂 wxasr_async::AsrHandle::push
}))?;
// drop(capture) 或 capture.stop() 停止
```

`start(device: Option<&str>, sink)` 可按名选设备；`devices()` 列出可用项。
`sink` 在音频线程上执行，必须快速返回（转发给通道即可）。

## 工具

```powershell
# loopback 通路探测（播放 440Hz 3 秒并统计采集）
cargo run --release --example probe -- 3

# 桌面音频 → 识别 → 每句一行 txt（--play 自测用，会播放指定 wav）
cargo run --release --example desktop_txt -- <模型目录> out.txt `
    --play ../wxasr/tests/data/tts_short_1.wav --secs 15
```

`desktop_txt` 捕获的是整个系统混音——机器上同时在放的音乐/视频
也会被转写进去（这正是 loopback 的语义）。

## 实测（Windows / WASAPI）

- loopback：48kHz 8ch f32 采集速率精确匹配播放（384000 样本/s）
- E2E：播放 TTS → loopback 采集 → 识别 → 断句写 txt，
  播放内容在中间结果中逐字浮现；背景音与播放内容同被捕获

MIT License
