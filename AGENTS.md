# project-a Agent Guide

离线 ASR 推理工作区(Windows,PowerShell):微信输入法 `.xnet` 模型的 Rust 推理、统一异步层、灵感来源于 Hex 并重新实现/大幅扩展的 GGUF 推理栈。四个 crate:

- `wxasr/`:`.xnet` 离线 ASR 推理库。thiserror 错误枚举(`Io/Model/Container/Audio`),公共 API 稳定。
- `wxasr-provider/`:音频采集抽象(cpal 实现),thiserror `Error` 枚举。
- `wxasr-async/`:统一异步 ASR 层——专用识别线程 + tokio mpsc 事件桥,自身不识别。后端实现
  `AsrBackend`/`AsrSession`/`AsrEvents` trait;`start_with` 是通用入口,`start` 保持 wxasr 专用
  薄包装。第一个后端是 `backend_wxasr.rs`(RMS VAD、partial 节流、上下文轮换)。
- `gguf-asr/`:灵感来源于 Hex(`G:\codes\rust\hex`,只读参考)、在其架构上
  重新实现并大幅扩展的 GGUF 推理栈(原名 hex-asr,2026-11-19 更名;许可
  注意事项见红线):engine(专用线程调度、有序命令、deadline、流失败整段回退)→ local_transcriber →
  gguf_session → transcribe-cpp(crates.io `=0.4.1` + `vulkan` feature,**无** Anomaly fork
  patch;fork 独有 API 一律剥除)。`session.rs` 是它对统一层契约的适配;
  `vad.rs` 是流式 Silero VAD 包装(`SileroVad`,供后端做静音 endpointing);
  `gguf_meta.rs` 只读 GGUF 头 KV(`GgufMeta`/`gguf_meta`,不加载权重,加载前
  可知模型家族与是否自报流式);`diarize.rs` 是独立 Sortformer 说话人分离
  (`Diarizer` 出 speaker turns,`assign_speakers` 按时间戳归并;说话人切分
  `SpeakerSplit`:`split_transcript` 整段切 / `resplit_mixed_segments` 混合段
  重切 / `speaker_split_for(&GgufMeta)` 头部快猜,见「测试」节)。

## 红线

- `wxasr-async/src/policy.rs` 冻结:轮换行为零变化(用户拍板,默认策略已改
  为 `Punct{10s,20s}`,断言旧默认的存量测试 `default_is_auto_15s_20s` 已删)。
- 事件枚举 v2(2026-11-18 拍板改名,一次性豁免"只增不破"):`AsrEvent` =
  `Partial`/`Commit`/`End`,**此后只增不破**;`AsrHandle`/`Config`/`start` 等
  其余公共接口只增不破;新代码与它们冲突时改新代码一侧。
- 兼容方向:gguf-asr 适配 wxasr-async 契约,不是反过来。
- wxasr 与 wxasr-provider 保留 thiserror 枚举(库 crate 的公共 API);换 eyre 会破坏
  调用方按变体 match 的能力,已明确决策不做。
- `G:\codes\rust\hex` 只读参考;从 Hex 复制/改写的代码,许可问题由用户承担。
- 重定向丢弃输出一律 `/dev/null`,禁止 `>nul`(MSYS 会创建名为 `nul` 的普通文件)。
- workspace 已设 `resolver = "3"`(edition 2024 成员要求);profile 忽略 warning 是存量,不必处理。

## 错误处理

- `wxasr-async`、`gguf-asr`:eyre(0.6.x)。不用 color_eyre——都是库 crate,不装全局 handler。
- `wxasr`、`wxasr-provider`:thiserror 枚举;被上层 `?` 进 `eyre::Report` 时自动携带完整错误链。

## 构建环境(每条 cargo 前必设)

日常开发优先用根目录 `justfile`(`just --list`):`check`/`check-gguf`/`test-model`/
`live`/`live-gguf`——gguf 相关配方自带下面的四件套环境,不必手动设。

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
  (powershell/cargo/rustup/`gguf_asr-*` 测试二进制)释放锁,修好再跑。

## 测试

- 依赖真实 GGUF 模型的测试用环境变量 `GGUFASR_MODEL` 门控,未设则 skip;
  Silero VAD 测试用 `GGUFASR_VAD_MODEL`(未设则用模型目录默认副本,缺文件
  skip,加载走 CPU 不需要 Vulkan);说话人分离测试用 `GGUFASR_DIARIZE_MODEL`
  (Sortformer diarize 模型,设了才装配进共享引擎,`engine.diarizes()` 为真)。
- 模型文件:项目自有副本在 `G:\env\model\asr\`(live_gguf 的默认目录,
  `GGUF_MODELS_DIR` 可覆盖),其中 `parakeet-unified-en-0.6b-Q8_0.gguf`(731MB,
  sha256 与 Hex 目录 pin 一致)是**唯一支持 native 流式 partial 的模型**(仅英语,
  需 GPU;whisper/qwen 无 native 流,经 gguf 后端的伪流式获得完整事件行为);
  `diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf`(133MB)是独立 diarize 模型
  (≤4 说话人,英语训练,live_gguf 自动检测);原始文件(用户经 Hex 应用下载,
  **勿动**)仍在 `C:\Users\Administrator\AppData\Local\hex\models\`:
  - `whisper-large-v3-turbo-Q8_0.gguf`(845MB)
  - `Qwen3-ASR-0.6B-Q8_0.gguf`(811MB)
  - `silero-vad-v6.2-F32.gguf`(1.2MB,VAD)
- 测试 wav fixtures:`wxasr\tests\data`;wxasr-async 集成测试用仓库内的 `.xnet` 模型。
  `wxasr-async\tests\data`(gitignore,本地生成)放 gguf 轮换端到端的 SAPI 英语
  wav + 参考文本,缺文件则该测试 skip;重新合成:
  `powershell Add-Type -AssemblyName System.Speech;` 后用
  `SpeechSynthesizer.SetOutputToWaveFile(路径, SpeechAudioFormatInfo(16000, Sixteen, Mono))`
  朗读多句英文,同文本写 `.txt`(12 句 ≈36s,Punct{10s,20s} 可切 2 段)。
  `gguf-asr\tests\data\sortformer-2spk-mix.wav`(384KB,取自上游
  handy-computer/transcribe.cpp `samples/`)是真实双说话人混音,门控
  diarize 切分转写的验收测试。
- 预期基线:wxasr-async lib / integration 全绿(无 feature 19+6;`--features gguf`
  lib 38,含 backend_gguf 的 8 个流式轮换 + 7 个伪流式 + diarize 装配 5 个);
  gguf-asr lib 133 全绿(131 过 + 2 ignored 属存量;含 gguf_meta 4 个、
  diarize 26 个、术语表 4 个);
  tests/model.rs 13 个(GGUFASR_MODEL/GGUFASR_DIARIZE_MODEL
  各自门控);gguf_backend 集成测试 7 个(GGUFASR_MODEL 按模型名各自
  激活,其余 skip)。Vulkan 加载大模型要几十秒,timeout 放大。
- 说话人分离:Sortformer 出 speaker turns,库的归并契约是时间戳;qwen3-asr
  `max_timestamp_kind = NONE`(C++ `finalize_utterance` 恒单段 [0, 整段]),
  无从归并。定稿的说话人切分由 `SpeakerSplit` 控制(`Engine::transcribe_clip_turns`
  第三参):`Auto`(默认,引擎按 `finest_timestamps` 真值解析;加载前的
  头部快猜 = `speaker_split_for(&GgufMeta)`,无时间戳家族清单硬编码自
  transcribe-cpp 各 arch 的 capabilities.cpp)/`Pieces`(无时间戳模型整段
  切分转写 `split_transcript`:按 turns 切音频、切点吸附 ±250ms 最静 100ms
  窗、逐片转写,一次定稿多段多 speaker,分片失败整体降级)/`MixedSegments`
  (whisper/parakeet 保留自身分段与时间戳,仅区间内跨换人点的段经
  `resplit_mixed_segments` 切分重转写,片自带行优先、逐段降级)/`Off`(纯归并)。
  后端 builder:`GgufBackend::new(e).with_pseudo_streaming(..).with_diarize(..)
  .with_speaker_split(..)`——typestate(axum Router<S> 式),split 依赖
  diarize,`with_speaker_split` 只存在于 HasDiarize 状态,顺序错是编译
  错误;存量组合构造器全保留。引擎未装配 diarize 模型时 turns 恒空,
  split 无害无效。
- **whisper "死等"已修复(2026-11-19 定案,根因是我们提取时的 bug)**:
  `speech_detector::loaded_session` 的 match scrutinee 里 `lock_model()`
  的临时守卫活到 match 结束,第二支条件 `None if lock_model().is_none()`
  同线程重锁同一 mutex → 自死锁;MODEL 为 None(我们不设 VAD 路径)时
  必进该支,whisper 首次转写(唯一走 `whisper_audio`→`speech_bounds`
  的架构)100% 挂死,表现与 GPU/驱动/fork/参数/模型文件全部无关
  (两天误排查的教训:挂点"随打印移动"其实是打印位置不同地包住同一行
  死锁;绕过包装层的探针从不挂;Hex 应用不触发是它的 MODEL 恒为
  Some,走第一支)。修复 = 锁一次、复用绑定。whisper 门控已全部解除,
  `just test-model whisper` 恢复验收(GPU 正常;CPU+whisper 撞引擎
  deadline 属设计内,伪流式测试自动 skip CPU)。诊断探针保留在
  tests/native_probe.rs(`GGUFASR_NATIVE_PROBE`/`GGUFASR_RAW_PROBE`/
  `GGUFASR_RAW_BISECT=threads|timestamps|doublewarm|full|caps`)。
- `just test-model` 一律 `--test-threads=1`:多个 36s 实时会话并行会争抢同一
  GPU 引擎,输家的流 decode 落后 → 0 partial(假失败,不是回归)。

## 统一层契约摘要

- `AsrEvents::emit(&mut self, ev: AsrEvent) -> bool`:返回 `false` 表示消费端已关闭,
  后端停止后续计算、尽快收尾(再发也会被丢弃)。
- `AsrSession::push(&mut self, pcm: &[i16], out: &mut dyn AsrEvents)`:按
  `AudioIn` 所声明格式的 i16(交错,任意块大小);`finish` 定稿并结束
  (空文本不发事件)。
- `AsrBackend::open(&self, cfg: &Config, audio: AudioIn) -> eyre::Result<Box<dyn AsrSession>>`:
  在识别线程内调用;失败则会话立即结束,消费端看到空会话。
- 音频格式(`AudioIn { sample_rate, channels }`)由调用方在 `start_with` 声明,
  统一层不写死采样率:降混/重采样在各后端内部完成(`preprocess::InputConverter`,
  16 kHz 单声道直通)。后端专属配置走后端构造参数,不进 `Config`
  (`rotate`/`partial`/`partial_min_fresh` 两个后端都解释,`partial_min_fresh`
  仅 wxasr;`event_capacity`/`thread_name` 属统一层)。
- 事件载荷 `AsrText`:`content` 唯一必填;`language`/`speaker`/`segments`/
  `words` 按模型能力填,不支持为 `None`/空(wxasr 后端只填 `content`,
  gguf 后端直通富字段)。
- 事件模型 v2:会话中 `Partial`(投机中间结果,Commit/End 后从空开始);
  `Commit` = 一段定稿(稳定前缀提交 + 切断,由轮换产生,会话继续);
  `End` = 会话结束(finish/句柄全 drop)补定的最后一段,空文本不发、最多
  一个。完整转写 = 按序拼接所有 `Commit` 与 `End`;会话结束以通道关闭为准,
  `End` 不是关闭事件。事件序 `(Partial* Commit)* Partial* End?` → 通道关闭。
  两个后端都解释 `rotate`(`Never` 时 gguf 一次会话恰一个 `End`);gguf 后端
  的轮换 = 老流 finish 排空解码滞后 + 新流从零(native 流一次性),静音信号
  Silero VAD(缺省/失败退化能量门),`punct_ms` 取 committed 新增句末标点
  落地时刻(滞后 ≤~1s,切点 = 判定时刻,与 wxasr 的 `rotate_at_punct` 不同)。
- gguf 伪流式(`GgufBackend::pseudo_streaming`/`with_pseudo(PseudoOptions)`):
  离线模型(whisper/qwen,无 native 流)的完整事件行为——`open` 见
  `engine.streams()` 为假且伪流式开启时走 `PseudoSession`(能真流式则自动
  升级,拿 parakeet 跑不退化)。增长缓冲 + 每 `partial_every_ms`(默认 2s)
  整段重转写 → 变化文本当 `Partial`(**允许整体回改**,契约内);轮换切段
  处整段权威转写 → `Commit`;`finish` → `End`。`punct_ms` 在最近一次转写
  文本上近似;全静音段跳过重转写;转写失败只 warn 一次(CPU+whisper 超引擎
  deadline 属设计内)。整段转写阻塞期间音频排队不丢;Commit/End 后向延迟 =
  段长/RTF(GPU 上 20s 段 ≈1s),`hard_ms` 即段缓冲上界。
- wxasr-async 的 `gguf` feature 启用 `backend_gguf`(gguf-asr 后端),编译需要
  CMake + Vulkan SDK 环境与 gguf-asr 相同。
- 术语表(`GgufBackend::with_vocabulary(terms)`,任意状态可叠加,空 = 不用):
  每个 gguf 会话开启时把术语(优先级序)经 `Engine::set_vocabulary` 交给引擎,
  进到底层每次 run/stream 的 `RunOptions::vocabulary`。**按模型能力适配、
  不分家族**:生效与否以 `Engine::accepts_vocabulary()` 为准(底层
  `Feature::Vocabulary`,加载后读一次),不支持的模型忽略并告警;术语怎么用
  (提示词 / 重打分)由模型自己决定,wxasr 侧不写任何家族分支。已开的流保留
  开始时的术语,变更在该流结束后生效(不打断流:`drive_stream`/`drive_yielding`
  延后处理);GPU 失败换 CPU 重载时术语保留(存在 `ModelSpec.vocabulary`)。
  入库 trim、去空、去含 NUL 的项(底层是 C 字符串,容不下会让每次 run 报错)。
