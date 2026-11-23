# wxasr 系列 API 文档

四个 crate 组成的离线 ASR 工作区:微信输入法 `.xnet` 模型推理、音频采集、
统一异步识别层、GGUF 推理栈。本文档面向**使用者**,按模块分篇;行为契约
与设计决策以各 crate 源码注释和根目录 `AGENTS.md` 为准。

| 篇 | crate | 一句话 |
| --- | --- | --- |
| [wxasr.md](wxasr.md) | `wxasr` | `.xnet` 离线推理(模型/流式识别) |
| [wxasr-provider.md](wxasr-provider.md) | `wxasr-provider` | 音频采集抽象(麦克风 / Windows 回环) |
| [wxasr-async.md](wxasr-async.md) | `wxasr-async` | 统一异步识别层(**大多数应用的入口**) |
| [gguf-asr.md](gguf-asr.md) | `gguf-asr` | GGUF 推理栈(引擎/流式会话/VAD/说话人分离) |
| [recipes.md](recipes.md) | — | 接线配方:live_gguf 模式、双会话共享引擎、构建与测试 |

## 我该用哪个?

- **写应用(转写/听写/会议记录)**:只用 `wxasr-async` 的
  `start_with` + 某个 `AsrBackend`(wxasr 模型用 `start`;GGUF 模型用
  `backend_gguf::GgufBackend`),配 `wxasr-provider` 采集。其余 crate
  的 API 不需要直接碰。
- **采集音频**:`wxasr-provider` 的 `Microphone` / `DesktopAudio`
  (输出恒为 16 kHz 单声道 i16)。
- **离线转一个文件**(不经统一层):`gguf_asr::Engine` 的
  `foreground` / `transcribe_clip`;wxasr 侧是 `Model::transcribe_wav`。
- **说话人分离单独用**:`gguf_asr::diarize`(`Diarizer` +
  `split_transcript` / `resplit_mixed_segments`)。
- **读 GGUF 头判断模型家族**:`gguf_asr::gguf_meta`(不加载权重,
  毫秒级)。

## 统一层事件模型(全后端一致,务必先读)

一个识别会话的生命周期由三类事件描述,经 tokio mpsc 通道送达消费端:

```
(Partial* Commit)* Partial* End?  →  通道关闭
```

- `Partial(AsrText)`:投机中间结果,**可整体回改**;每次 `Commit`/`End`
  后从空开始。
- `Commit(AsrText)`:一段定稿(轮换产生),会话继续。
- `End(AsrText)`:会话结束(`finish`/句柄全 drop)补定的最后一段;
  **空文本不发、最多一个**。会话是否结束以**通道关闭**(`recv` 返回
  `None`)为准,`End` 不是关闭事件。
- 完整转写 = 按序拼接所有 `Commit` 与 `End`。

事件载荷 `AsrText`:`content` 唯一必填;`language` / `speaker` /
`segments` / `words` 按模型能力填(不支持即 `None`/空)。说话人语义:
顶层 `speaker` 仅当整段同一人;混合多人时为 `None`、各分段
(`segments[].speaker`)各带;`Partial` 永不带 speaker。
`segments` 的时间戳是**会话绝对时刻**(自会话起点累计,轮换段已平移)。
