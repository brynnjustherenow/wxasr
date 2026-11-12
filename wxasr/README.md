# wxasr

微信输入法离线语音识别模型（`.xnet`）的纯 Rust 推理库。无深度学习框架依赖，
int8 动态量化 GEMM 自带 AVX2 内核，Windows/Linux/macOS/Android 均可编译。

## 能力

- **一次性转写**：`Model::transcribe_wav` 整段解码（16kHz 单声道 16bit wav）
- **伪流式识别**：`Model::stream` 边说边出中间结果，已定稿前缀不变
- **低资源占用**：权重 mmap 零拷贝加载（~5ms），int8 推理，4 线程封顶

## 实测（i7 / Windows / release）

| 场景 | 结果 |
|---|---|
| 4.6s 真人语音 | 191ms，实时率 0.04 |
| 36.1s 长音频整段 | 2.2s，实时率 0.06，CER 3.6%~5.8%（TTS 对拍） |
| 36.1s 流式（90 个中间结果） | 全程 13.3s，实时率 0.37 |
| 短句 TTS 对拍 | CER 0% ~ 5.3% |

## 模型文件

本库不提供、也不指向任何模型下载来源。请自行获取微信输入法的离线语音包
（apk 格式，双层 zip），解出以下两个文件放入同一目录：

```
model/
├── embed_140m_...dyQu.onnx.xnet   # 任一 *.xnet
└── dict.decoder.utf8.txt          # CTC 词表
```

## 用法

```rust
let model = wxasr::Model::load("model")?;      // Arc<Model>，加载 ~5ms
let text = model.transcribe_wav("a.wav")?;      // 一次性转写

let mut s = model.stream();                     // 流式
s.push(&pcm);                                   // 16kHz 单声道 i16，任意块
let partial = s.partial();                      // 中间结果（按 s.fresh() 节流）
let final_text = s.finish();                    // 定稿
```

命令行转写：`cargo run --release --example cli -- <模型目录> a.wav`
模型分析：`cargo run --release --example dump -- <模型目录>`

## 测试

```powershell
# 生成 TTS 测试音频（需 zh-CN 语音，如 Microsoft Huihui）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gen_tts.ps1
# 单测 + 集成测试（真实模型 + 音频缺失时自动跳过）
cargo test --release
```

模型目录默认取 `WXASR_MODEL` 环境变量，缺省 `G:/codes/rust/project-a/model`。

## 架构

```
src/
├── xnet.rs    容器解析（尾部记录表扫描、dtype/量化 scale/形状校验）
├── fbank.rs   40 维 HTK log-mel + 对数能量，50 帧滑动均值 CMS
├── kernels.rs 激活 per-row 动态量化 + int8 GEMM（AVX2 2×4 微内核 / 标量兜底）
├── model.rs   conv 前端(10ms→50ms) + 40 层 pre-norm Transformer + CTC
├── stream.rs  block_spec=100:400:100 分块定稿、双 blank 截断、静音 pad
└── lib.rs     公开 API
```

流式参数取自官方 `sr_online.conf`（`block_spec = 100:400:100`）；
特征归一化经实测采用「只减最近 50 帧滑动均值、不除方差」。

## 致谢与许可

本库的容器格式、特征参数与流式策略参考了
[J3n5en/voicekey](https://github.com/J3n5en/voicekey) 中 `wtlocal` 的实现思路。
由于该项目未附带开源许可证，本库未复用其任何代码，
全部为参照其行为与官方模型包内配置的独立重实现。

- 本 crate 代码原创实现，MIT
- 本库与微信/腾讯无关联；`.xnet` 模型与词表版权归其所有者所有，
  请自行获取并仅作个人学习研究使用，请勿再分发
