//! HTK 风格流式 FBank 特征：39 维 log-mel + 对数能量（共 40 维）。
//!
//! 参数：16kHz 单声道，25ms 窗 / 10ms 帧移，512 点 FFT（补零），
//! pre-emphasis 0.97，Hamming 窗。归一化只做「减去最近 0.5 秒的
//! 滑动均值」（对应官方 `cms_win=50`），不除方差——官方 CMS 表的
//! 方差缩放对底噪高、起伏小的麦克风音频会压扁动态，导致大量漏字。

use std::collections::VecDeque;
use std::sync::OnceLock;

/// 特征维度：39 mel + 1 对数能量
pub const DIM: usize = 40;

const WIN: usize = 400; // 25ms
const HOP: usize = 160; // 10ms
const NFFT: usize = 512;
const NCH: usize = 39; // mel 通道数
/// 滑动均值窗口（帧），对应官方配置 cms_win=50
const CMS_WIN: usize = 50;

struct Tables {
    ham: Vec<f64>,
    /// FFT 旋转因子（共 NFFT/2 个）
    tw: Vec<(f64, f64)>,
    /// 每个功率谱频点 k（对应 FFT bin k+1）的通道分摊：(低通道, 低权重, 高通道, 高权重)
    mel: Vec<(usize, f64, usize, f64)>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let pi = std::f64::consts::PI;
        let ham = (0..WIN)
            .map(|n| 0.54 - 0.46 * (2.0 * pi * n as f64 / (WIN - 1) as f64).cos())
            .collect();
        let tw = (0..NFFT / 2)
            .map(|i| {
                let a = -2.0 * pi * i as f64 / NFFT as f64;
                (a.cos(), a.sin())
            })
            .collect();
        // 40 个中心点在 mel 域 0→mel(8000) 均分，构成 39 个通道的边界
        let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
        let hi = mel(8000.0);
        let cf: Vec<f64> = (0..NCH + 1)
            .map(|i| hi * (i + 1) as f64 / (NCH + 1) as f64)
            .collect();
        let bins = (1..=NFFT / 2)
            .map(|k| {
                let m = mel(k as f64 * 16000.0 / NFFT as f64);
                // 中心点中小于 m 的个数，决定落点
                let c = cf.iter().take_while(|&&v| v < m).count();
                match c {
                    0 => (0, m / cf[0], 0, 0.0),
                    c if c < NCH => {
                        let w = (cf[c] - m) / (cf[c] - cf[c - 1]);
                        (c - 1, w, c, 1.0 - w)
                    }
                    c if c == NCH => (NCH - 1, (cf[NCH] - m) / (cf[NCH] - cf[NCH - 1]), 0, 0.0),
                    _ => (0, 0.0, 0, 0.0), // 高于最后一个中心点：不分摊
                }
            })
            .collect();
        Tables { ham, tw, mel: bins }
    })
}

/// 原地 radix-2 迭代 FFT
fn fft(re: &mut [f64], im: &mut [f64], tw: &[(f64, f64)]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // 位反转置换
    let mut j = 0;
    for i in 1..n - 1 {
        let mut b = n >> 1;
        while j & b != 0 {
            j ^= b;
            b >>= 1;
        }
        j |= b;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    // 蝶形
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for base in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = tw[k * step];
                let (p, q) = (base + k, base + k + len / 2);
                let (tr, ti) = (re[q] * wr - im[q] * wi, re[q] * wi + im[q] * wr);
                re[q] = re[p] - tr;
                im[q] = im[p] - ti;
                re[p] += tr;
                im[p] += ti;
            }
        }
        len <<= 1;
    }
}

/// 由一个 400 样本窗算出一帧特征。
/// 能量用原始样本计算；mel 部分经过 pre-emphasis + Hamming + FFT。
fn frame(x: &[i16]) -> [f64; DIM] {
    let t = tables();
    let (mut re, mut im) = ([0.0f64; NFFT], [0.0f64; NFFT]);
    let mut energy = 0.0;
    for n in 0..WIN {
        let v = x[n] as f64;
        energy += v * v;
        let pe = if n == 0 {
            v * (1.0 - 0.97)
        } else {
            v - 0.97 * x[n - 1] as f64
        };
        re[n] = pe * t.ham[n];
    }
    fft(&mut re, &mut im, &t.tw);
    let mut out = [0.0f64; DIM];
    for (k, &(lo, wl, hi, wh)) in t.mel.iter().enumerate() {
        let p = re[k + 1] * re[k + 1] + im[k + 1] * im[k + 1];
        out[lo] += p * wl;
        out[hi] += p * wh;
    }
    for v in &mut out[..NCH] {
        *v = v.max(1.0).ln();
    }
    out[NCH] = energy.max(1e-10).ln();
    out
}

/// 流式特征器：喂 PCM，产出减去滑动均值后的 40 维帧。
pub struct Fbank {
    buf: Vec<i16>,
    hist: VecDeque<[f64; DIM]>,
    sum: [f64; DIM],
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Fbank {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            hist: VecDeque::with_capacity(CMS_WIN),
            sum: [0.0; DIM],
        }
    }

    /// 当前窗口均值（窗口未满时按已有帧数算）
    fn mean(&self) -> [f64; DIM] {
        let n = self.hist.len().max(1) as f64;
        std::array::from_fn(|i| self.sum[i] / n)
    }

    fn normalize(&self, f: &[f64; DIM]) -> [f32; DIM] {
        let m = self.mean();
        std::array::from_fn(|i| (f[i] - m[i]) as f32)
    }

    /// 喂入一段 PCM（16kHz 单声道 i16），产出的帧追加到 `out`
    pub fn push(&mut self, pcm: &[i16], out: &mut Vec<[f32; DIM]>) {
        self.buf.extend_from_slice(pcm);
        let mut p = 0;
        while p + WIN <= self.buf.len() {
            let f = frame(&self.buf[p..p + WIN]);
            if self.hist.len() == CMS_WIN {
                let old = self.hist.pop_front().unwrap();
                for i in 0..DIM {
                    self.sum[i] -= old[i];
                }
            }
            self.hist.push_back(f);
            for i in 0..DIM {
                self.sum[i] += f[i];
            }
            out.push(self.normalize(&f));
            p += HOP;
        }
        self.buf.drain(..p);
    }

    /// 按当前滑动均值归一化的「数字静音」帧。
    /// 注意：归一化空间的全 0 帧并不是静音（mel 值为 0 意味着功率 1），
    /// 静音应是 mel=0、能量=ln(1e-10)。
    pub fn silence(&self) -> [f32; DIM] {
        let mut f = [0.0f64; DIM];
        f[NCH] = 1e-10f64.ln();
        self.normalize(&f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_frame_is_log_domain_zero() {
        let fb = Fbank::new();
        let s = fb.silence();
        // 未喂入任何音频时均值为 0：静音帧 = 原始 log 域值
        assert!((s[NCH] - 1e-10f64.ln() as f32).abs() < 1e-4);
        for v in &s[..NCH] {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn periodic_signal_converges_to_zero_mean() {
        // 周期 160 的信号使每个窗内容相同；均值收敛后输出应恰为 0
        let pcm: Vec<i16> = (0..8400)
            .map(|n| {
                (12000.0 * (2.0 * std::f64::consts::PI * (n % 160) as f64 / 160.0).sin()) as i16
            })
            .collect();
        let mut fb = Fbank::new();
        let mut out = Vec::new();
        fb.push(&pcm, &mut out);
        assert!(out.len() >= 50, "应有至少 50 帧，实际 {}", out.len());
        for f in &out[49..] {
            for v in f {
                assert!(v.abs() < 1e-5, "收敛后帧应近全零: {f:?}");
            }
        }
    }

    #[test]
    fn frame_count_matches_hop() {
        let mut fb = Fbank::new();
        let mut out = Vec::new();
        fb.push(&vec![0i16; 9999], &mut out);
        assert_eq!(out.len(), (9999 - WIN) / HOP + 1);
    }

    #[test]
    fn one_khz_tone_peaks_near_expected_mel_channel() {
        // 1kHz 正弦（周期 16 样本）：mel 域位置 u = mel(1000)/mel(8000)*40 ≈ 14.1。
        // 注意要用未归一化的原始帧：周期信号经滑动均值归一化后谱形会被抵消
        let pcm: Vec<i16> = (0..8400)
            .map(|n| {
                (8000.0 * (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 16000.0).sin()) as i16
            })
            .collect();
        let f = frame(&pcm[0..WIN]);
        let peak = (0..NCH)
            .fold(
                (0usize, f64::MIN),
                |b, i| if f[i] > b.1 { (i, f[i]) } else { b },
            )
            .0;
        // pre-emphasis 会把能量重心略微上移
        assert!((12..=15).contains(&peak), "1kHz 峰值通道 {peak}");
    }
}
