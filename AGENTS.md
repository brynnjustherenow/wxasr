# project-a Agent Guide

离线 ASR 推理工作区(Windows,PowerShell):微信输入法 `.xnet` 模型的 Rust 推理、统一异步层、从 Hex 提取的 GGUF 推理栈。四个 crate:

- `wxasr/`:`.xnet` 离线 ASR 推理库。thiserror 错误枚举(`Io/Model/Container/Audio`),公共 API 稳定。
- `wxasr-provider/`:音频采集抽象(cpal 实现),thiserror `Error` 枚举。
- `wxasr-async/`:统一异步 ASR 层——专用识别线程 + tokio mpsc 事件桥,自身不识别。后端实现
  `AsrBackend`/`AsrSession`/`AsrEvents` trait;`start_with` 是通用入口,`start` 保持 wxasr 专用
  薄包装。第一个后端是 `backend_wxasr.rs`(RMS VAD、partial 节流、上下文轮换)。
- `hex-asr/`:Hex(`G:\codes\rust\hex`,只读参考)GGUF 推理栈的提取:
  engine(专用线程调度、有序命令、deadline、流失败整段回退)→ local_transcriber →
  gguf_session → transcribe-cpp(crates.io `=0.4.1` + `vulkan` feature,**无** Anomaly fork
  patch;fork 独有 API 一律剥除)。`session.rs` 是它对统一层契约的适配。

## 红线

- `wxasr-async/src/policy.rs` 冻结:零改动,轮换行为零变化。其存量测试失败
  `policy::tests::default_is_auto_15s_20s` 保持原样,不要修。
- wxasr-async 原有公共接口(`AsrEvent`/`AsrHandle`/`Config`/`start`)只增不破;
  新代码与它们冲突时改新代码一侧。
- 兼容方向:hex-asr 适配 wxasr-async 契约,不是反过来。
- wxasr 与 wxasr-provider 保留 thiserror 枚举(库 crate 的公共 API);换 eyre 会破坏
  调用方按变体 match 的能力,已明确决策不做。
- `G:\codes\rust\hex` 只读参考;从 Hex 复制/改写的代码,许可问题由用户承担。
- 重定向丢弃输出一律 `/dev/null`,禁止 `>nul`(MSYS 会创建名为 `nul` 的普通文件)。
- workspace 的 profile/resolver warning 是存量,不必处理。

## 错误处理

- `wxasr-async`、`hex-asr`:eyre(0.6.x)。不用 color_eyre——都是库 crate,不装全局 handler。
- `wxasr`、`wxasr-provider`:thiserror 枚举;被上层 `?` 进 `eyre::Report` 时自动携带完整错误链。

## 构建环境(每条 cargo 前必设)

```powershell
$env:Path = "C:\Program Files\CMake\bin;$env:Path"
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.363.0"
$env:LIB = "C:\VulkanSDK\1.4.363.0\Lib;$env:LIB"
$env:INCLUDE = "C:\VulkanSDK\1.4.363.0\Include;$env:INCLUDE"
```

- 漏 `LIB` 会 LNK1181(找不到 vulkan-1.lib)。装完工具链的已存进程拿不到机器环境变量,
  必须显式设。
- transcribe-cpp-sys 的 C++ 部分首次或依赖树变动后要编几分钟;长 cargo 命令 timeout
  ≥600000ms,或后台跑 + 输出重定向到日志文件再读(海量编译日志会堵 shell)。
- 并发 cargo 按构建锁串行,排队等待属正常,不要当成卡死。
- 测试 runner 无超时:挂起的测试会永久占构建锁。测试挂死时按命令行匹配杀进程树
  (powershell/cargo/rustup/`hex_asr-*` 测试二进制)释放锁,修好再跑。

## 测试

- 依赖真实 GGUF 模型的测试用环境变量 `HEXASR_MODEL` 门控,未设则 skip。
- 模型文件(用户经 Hex 应用下载,**勿动**):`C:\Users\Administrator\AppData\Local\hex\models\`
  - `whisper-large-v3-turbo-Q8_0.gguf`(845MB)
  - `Qwen3-ASR-0.6B-Q8_0.gguf`(811MB)
  - `silero-vad-v6.2-F32.gguf`(1.2MB,VAD)
- 测试 wav fixtures:`wxasr\tests\data`;wxasr-async 集成测试用仓库内的 `.xnet` 模型。
- 预期基线:wxasr-async lib 唯一失败是 policy.rs 存量失败,integration 全绿;hex-asr 全绿。
  Vulkan 加载大模型要几十秒,timeout 放大。

## 统一层契约摘要

- `AsrEvents::emit(&mut self, ev: AsrEvent) -> bool`:返回 `false` 表示消费端已关闭,
  后端停止后续计算、尽快收尾(再发也会被丢弃)。
- `AsrSession::push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents)`:16kHz 单声道
  i16,任意块大小;`finish` 定稿并结束(空文本不发事件)。
- `AsrBackend::open(&self, cfg: &Config) -> eyre::Result<Box<dyn AsrSession>>`:在识别
  线程内调用;失败则会话立即结束,消费端看到空会话。
- 后端专属配置走后端构造参数,不进 `Config`(`rotate`/`partial`/`partial_min_fresh`
  由 wxasr 后端解释,其他后端可忽略;`event_capacity`/`thread_name` 属统一层)。
- 事件模型:会话中 `Partial`(投机中间结果),定稿 `Final`;一次 hex 会话恰一个 Final
  (Hex 语义无轮换),多段 Final 是 wxasr 后端自己的行为。会话结束以通道关闭为准,
  `Final` 不是关闭事件。
