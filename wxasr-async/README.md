# wxasr-async

[wxasr](../wxasr) 的异步事件流封装：一个专用识别线程拥有 `Stream`
（中间结果节流 + 静音断句），tokio mpsc 桥接事件，async 侧只做分发。

## 用法

```rust
use wxasr_async::{start, AsrEvent, Config};

let model = wxasr::Model::load("model")?;
let (handle, mut rx) = start(model, Config::default());

handle.push(&pcm);   // 采集回调里调用：同步、非阻塞、不丢音频
handle.finish();     // 结束识别

while let Some(ev) = rx.recv().await {
    match ev {
        AsrEvent::Partial(t) => { /* 边说边显（已按计算成本自动节流） */ }
        AsrEvent::Sentence(t) => { /* 静音超 silence_ms 自动断句定稿 */ }
        AsrEvent::Final(t) => { println!("{t}"); break; } /* 关闭事件 */
    }
}
```

## 配置

```rust
Config {
    silence_ms: 800,        // 断句静音窗口；0 = 不自动断句
    partial: true,          // 是否产生中间结果
    partial_min_fresh: 30,  // 中间结果最小新增帧数（10ms/帧）
    event_capacity: 64,     // 事件通道容量
    ..Default::default()
}
```

## 静音检测（内置 VAD）

- 判定用 25ms 子块 RMS **峰值**：语音有音节起伏、平稳底噪没有，
  因此能区分「软语音 vs 响底噪」（实测 TTS 底噪 RMS≈633、软语音
  均值仅 2 倍于底噪，均值判定会误判，峰值判定不会）
- 噪声底非对称跟踪：静音段快速下跟、极慢上漂，杜绝被软语音
  棘轮式抬高的恶性循环
- 注意：噪声底初值取自首个块——若用户开口极快，请先喂一小段
  环境音再开始正式采集

## 线程模型

```
采集回调(同步) ──std mpsc(无界,不丢)──▶ wxasr-worker 线程 ──tokio mpsc(有界)──▶ async 消费
```

识别线程默认名 `wxasr-<uuidv7>`（自动唯一，多流并发不重名），可用
`Config::thread_name` 显式命名（如 `wxasr-desktop`）；线程随 `finish()`
或所有句柄 drop 而退出；
CPU 密集的前向计算在 worker 内的 wxasr 专用线程池（≤4 线程）上执行，
async 执行器不被阻塞。

## 测试

```powershell
cargo test --release   # 需真实模型 + ../wxasr/tests/data 音频（缺失自动跳过）
```

MIT License
