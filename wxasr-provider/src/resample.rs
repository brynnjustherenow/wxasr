//! 流式线性重采样：任意采样率 → 目标采样率。
//!
//! 对语音识别足够（fbank 只到 39 个 mel 通道），无需窗函数滤波；
//! 零依赖、状态连续，适合在音频回调里逐块使用。

/// 状态保持的线性重采样器。`push` 接收单声道 f32（-1..1），
/// 追加产出目标采样率的 i16。
pub struct Resampler {
    step: f64, // 输入/输出采样率之比
    t: f64,    // 下一个输出点在 (上一输入样本, 当前输入样本) 区间内的位置
    last: f32, // 上一个输入样本
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        Self { step: in_rate as f64 / out_rate as f64, t: 0.0, last: 0.0 }
    }

    pub fn push(&mut self, samples: &[f32], out: &mut Vec<i16>) {
        let mut prev = self.last;
        for &s in samples {
            while self.t < 1.0 {
                let v = prev + (s - prev) * self.t as f32;
                out.push((v.clamp(-1.0, 1.0) * 32767.0) as i16);
                self.t += self.step;
            }
            self.t -= 1.0;
            prev = s;
        }
        self.last = prev;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_count_matches_ratio() {
        // 48k → 16k：1/3 数量；跨块状态连续
        let mut r = Resampler::new(48000, 16000);
        let input: Vec<f32> = (0..4800)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let mut out = Vec::new();
        for chunk in input.chunks(713) { // 故意不整除，考验状态
            r.push(chunk, &mut out);
        }
        assert!((out.len() as i64 - 1600).abs() <= 2, "len={}", out.len());
    }

    #[test]
    fn dc_signal_passes_through() {
        let mut r = Resampler::new(44100, 16000);
        let mut out = Vec::new();
        r.push(&[0.5f32; 1000], &mut out);
        // 首个输出是启动瞬态（t=0 取初始 last=0），其后应为恒定 0.5
        for v in out.iter().skip(1) {
            assert!(((v - 16383) as i32).abs() <= 2, "v={v}");
        }
    }

    #[test]
    fn upsampling_doubles_count() {
        let mut r = Resampler::new(8000, 16000);
        let mut out = Vec::new();
        r.push(&vec![0.0f32; 800], &mut out);
        assert!((out.len() as i64 - 1600).abs() <= 2);
    }

    #[test]
    fn sine_frequency_preserved() {
        // 440Hz 持续 1 秒：过零数是时间量，与采样率无关，
        // 48k→16k 后过零数应基本不变（频率未被改变）
        let mut r = Resampler::new(48000, 16000);
        let input: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        let mut out = Vec::new();
        r.push(&input, &mut out);
        let crossings = |x: &[i16]| {
            x.windows(2).filter(|w| (w[0] >= 0) != (w[1] >= 0)).count()
        };
        let expect = crossings(&input.iter().map(|&v| (v * 32767.0) as i16).collect::<Vec<_>>());
        let got = crossings(&out);
        assert!((got as i64 - expect as i64).abs() < 8, "{got} vs {expect}");
    }
}
