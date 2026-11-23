# wxasr — `.xnet` 离线推理

微信输入法 `.xnet` 模型的 Rust 推理库。纯 CPU、同步 API,无外部依赖。
错误是 thiserror 枚举(`Io / Model / Container / Audio`),公共 API 稳定。

```rust
use std::sync::Arc;
let model = wxasr::Model::load("path/to/model_dir")?; // 目录内需有 .xnet 文件;返回 Arc<Model>
```

## Model(`src/model.rs`)

```rust
pub struct Model { .. }

impl Model {
    /// 从目录加载(mmap 权重;目录中找第一个 .xnet 文件)。多线程共享:直接 clone Arc。
    pub fn load(dir: impl AsRef<Path>) -> Result<Arc<Model>>;

    /// 整个 wav 文件一次性转写(16 kHz 单声道最佳;内部做 fbank)。
    pub fn transcribe_wav(&self, path: impl AsRef<Path>) -> Result<String>;

    /// 从当前模型开一条同步流式识别(见下)。Arc 方法。
    pub fn stream(self: &Arc<Self>) -> Stream;
}
```

## Stream — 同步流式识别(`src/stream.rs`)

wxasr 后端轮换与 partial 的底层实现;一般应用不直接用它,走
`wxasr-async` 的 `start`。直接使用时注意:**按实时节奏喂音频才会产生
partial**(快于实时的灌入不产生中间结果)。

```rust
let mut stream = model.stream();
stream.push(&pcm_i16);          // 任意块大小,16 kHz 单声道 i16
let text: String = stream.partial();   // 投机中间结果(可回改)
let fresh = stream.fresh();            // 自上次 partial 以来的样本数(节流用)

let final_text = stream.finish();      // 消耗 self,定稿整段

// 轮换:定稿当前段并立刻继续新段(不结束会话)
let (committed, mut next) = stream.rotate();
let (committed2, next2) = stream.rotate_at_punct(); // 在最近的句末标点处切
let punct_at: Option<u32> = stream.punct_ms();      // committed 新增句末标点的时刻
```

## 其它导出

- `wxasr::Error` / `wxasr::Result`:thiserror 错误枚举。
- `wxasr::Fbank` / `wxasr::DIM`:fbank 特征(内部用;测试/对拍才碰)。
