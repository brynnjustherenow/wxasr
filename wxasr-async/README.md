# wxasr-async

[wxasr](../wxasr) 的异步事件流封装：一个专用识别线程拥有 `Stream`
（中间结果节流 + 上下文轮换定稿），tokio mpsc 桥接事件，async 侧只做分发。

## 用法

```rust
use wxasr_async::{start, AsrEvent, Config};

let model = wxasr::Model::load("model")?;
let (handle, mut rx) = start(model, Config::default());

handle.push(&pcm);   // 采集回调里调用：同步、非阻塞、不丢音频
handle.finish();     // 结束识别

let mut transcript = String::new();
while let Some(ev) = rx.recv().await {
    match ev {
        AsrEvent::Partial(t) => { /* 当前段边说边显 */ }
        AsrEvent::Final(t) => transcript += &t, // 一段定稿（轮换或结束时）
    }
}
// 通道关闭 = 会话结束，transcript 即完整转写
```

## 配置

```rust
Config {
    rotate: RotatePolicy::Auto { soft_ms: 15_000, hard_ms: 20_000 }, // 上下文轮换策略
    partial: true,          // 是否产生中间结果
    partial_min_fresh: 30,  // 中间结果最小新增帧数（10ms/帧）
    event_capacity: 64,     // 事件通道容量
    ..Default::default()
}
```

## 上下文轮换

`wxasr::Stream` 的内部状态（特征帧缓冲约 16KB/s、已定稿 token）随会话
时长线性增长，长会话会持续挤压内存并拖慢中间结果。worker 按策略在合适
时机定稿当前流、发出一个 `Final`（该段定稿文本）并接续。每次 `Final` 后
`Partial` 从空开始（取决于策略）；worker 不保留已定稿文本，资源占用与会话总长无关。
完整转写 = 顺序拼接所有 `Final`。

- `Auto { soft_ms, hard_ms }`（默认 15s/20s）：软上限到达后于下一次
  ≥500ms 静音处轮换；硬上限到达则说话中也强制轮换。
- `Punct { soft_ms, hard_ms }`：标点触发——软上限到达后，出现下一个
  已定稿标点即轮换，切点取在标点之后。标点是模型自己的断句信号，
  `Final` 以 `。，？` 等收尾，适合「一句一行」的上屏(无特殊需求推荐使用这个)；
- `Never`：永不轮换，状态随会话线性增长
- `Custom(Arc<dyn Fn(&RotateCtx) -> bool>)`：自定义判定，依据为自上次
  轮换以来的音频毫秒数、尾部连续静音毫秒数、最近定稿标点的位置

## 静音检测（内置 VAD）

- 判定用 25ms 子块 RMS **峰值**：语音有音节起伏、平稳底噪没有，
  因此能区分「软语音 vs 响底噪」（实测 TTS 底噪 RMS≈633、软语音
  均值仅 2 倍于底噪，均值判定会误判，峰值判定不会）
- 噪声底非对称跟踪：静音段快速下跟、极慢上漂，杜绝被软语音
  棘轮式抬高的恶性循环
- 注意：噪声底初值取自首个块——若用户开口极快，请先喂一小段
  环境音再开始正式采集
- VAD 不直接面向用户，它为轮换策略提供静音信号

## 线程模型

```
采集回调(同步) ──std mpsc(无界)──▶ wxasr-worker 线程 ──tokio mpsc(有界)──▶ async 消费
```

识别线程默认名 `wxasr-<uuidv7>`，可用
`Config::thread_name` 显式命名（如 "wxasr-desktop"）；线程随 `finish()`
或所有句柄 drop 而退出；
CPU 密集的前向计算在 worker 内的 wxasr 专用线程池（≤4 线程）上执行，
async 执行器不被阻塞。

## 测试

```powershell
cargo test --release   # 需真实模型 + ../wxasr/tests/data 音频（缺失自动跳过）
```

MIT License
