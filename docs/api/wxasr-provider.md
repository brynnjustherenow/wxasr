# wxasr-provider — 音频采集抽象

cpal 之上的采集层。**所有实现的输出恒为 16 kHz 单声道 i16**(内部完成
下混与重采样),正好直接对接统一层 `AudioIn::MONO_16K`。错误为
thiserror 枚举(`Device / Unsupported`)。

```rust
use wxasr_provider::{AudioSource, Capture, DesktopAudio, Microphone, PcmSink};

pub trait AudioSource: Send + Sync {
    fn name(&self) -> &'static str;                              // "microphone" / "desktop"
    fn devices(&self) -> Result<Vec<String>>;                    // 设备名列表
    fn start(&self, device: Option<&str>, sink: PcmSink) -> Result<Capture>;
    // device = None 用默认设备;Some(名称) 按名精确匹配
}

pub type PcmSink = Arc<dyn Fn(&[i16]) + Send + Sync>;
```

## 两个实现

| 实现 | 说明 |
| --- | --- |
| `Microphone::new()` | 输入设备(默认输入配置)。 |
| `DesktopAudio::new()` | **系统回环**(正在播放的声音)。Windows 上是 WASAPI loopback:在**输出设备**上建输入流,无需虚拟声卡;Linux 找名含 `monitor` 的输入源;macOS 需要 BlackHole 之类虚拟设备,否则 `Error::Unsupported`。 |

## Capture — 采集会话

```rust
pub struct Capture { .. }
impl Capture {
    pub fn stop(&mut self);        // 幂等
    pub fn is_running(&self) -> bool;
}
// drop 自动停止
```

## 使用要点

- **sink 回调在 cpal 音频线程上执行**:只做转发(通道/环形缓冲/写盘),
  必须快速返回,不要做识别或阻塞操作。
- 麦克风与回环是两条独立流,各 `start` 一次即可并存(双路监听)。
- 采样率/声道无需关心:任何设备原生格式都会被规整到 16k/mono/i16。
- 落盘:在 sink 里 tee 一份给 `hound::WavWriter`(spec = 1ch/16k/i16)
  即与数据格式一致。
