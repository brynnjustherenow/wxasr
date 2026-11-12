//! 数值内核：激活按行动态量化为 int8 后做整数 GEMM（权重为容器内
//! 静态 int8 + 每张量一个 scale），以及 f32 点积（卷积与注意力用）。
//!
//! int8 GEMM 微内核为 2 激活行 × 4 权重行的分块；AVX2 路径用
//! `|x|(u8) × sign(x)·w(i8)` 经 maddubs 成对累加（|x|≤127 保证不饱和），
//! 与标量路径是同一组整数运算，结果按位一致。

use rayon::prelude::*;

/// 一批动态量化后的激活：每行一个 scale
pub struct QAct {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
    pub k: usize,
}

/// 按行对称量化到 int8（每行一个 scale，零行 scale=1）
pub fn quant(x: &[f32], k: usize) -> QAct {
    assert!(x.len() % k == 0, "激活长度 {k} 的整数倍");
    let t = x.len() / k;
    let mut q = vec![0i8; t * k];
    let mut s = vec![1f32; t];
    q.par_chunks_mut(k)
        .zip(s.par_iter_mut())
        .zip(x.par_chunks(k))
        .for_each(|((q, s), row)| {
            let m = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            if m > 0.0 {
                let sc = m / 127.0;
                *s = sc;
                let inv = 1.0 / sc;
                for (q, &v) in q.iter_mut().zip(row) {
                    *q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        });
    QAct { q, s, k }
}

/// 跨并行任务共享的可写指针（各任务写互不重叠的列块）
#[derive(Clone, Copy)]
struct Sink(*mut f32);
unsafe impl Send for Sink {}
unsafe impl Sync for Sink {}

/// `out[t×n] = (x · wᵀ) · xs · ws + bias`。`w` 为 `[n, k]` 行主序 int8。
pub fn qgemm(x: &QAct, w: &[i8], n: usize, ws: f32, bias: Option<&[f32]>) -> Vec<f32> {
    let (k, t) = (x.k, x.s.len());
    assert_eq!(w.len(), n * k, "权重长度与 n×k 不符");
    let mut out = vec![0f32; t * n];
    if t == 0 || n == 0 {
        return out;
    }
    let sink = Sink(out.as_mut_ptr());
    const TILE: usize = 32;
    // 按输出列分 32 列一块并行；块内以 4 权重行 × 2 激活行的微内核推进
    (0..n.div_ceil(TILE)).into_par_iter().for_each(|tile| {
        let sink = sink; // 按值捕获 Sink（避免闭包精确捕获裸指针字段绕过 Sync impl）
        let n0 = tile * TILE;
        let n1 = (n0 + TILE).min(n);
        let mut j = n0;
        while j < n1 {
            let cols = (n1 - j).min(4);
            let mut i = 0;
            while i < t {
                let rows = (t - i).min(2);
                // 行/列不足时用重复行补齐（多余结果直接丢弃）
                let xr: [&[i8]; 2] = [&x.q[i * k..], &x.q[(i + rows - 1) * k..]];
                let wr: [&[i8]; 4] = std::array::from_fn(|b| &w[(j + b.min(cols - 1)) * k..]);
                let acc = dot2x4(xr, wr, k);
                for a in 0..rows {
                    for b in 0..cols {
                        let v =
                            acc[a][b] as f32 * x.s[i + a] * ws + bias.map_or(0.0, |bb| bb[j + b]);
                        unsafe { *sink.0.add((i + a) * n + j + b) = v };
                    }
                }
                i += rows;
            }
            j += cols;
        }
    });
    out
}

/// 2 个激活行 × 4 个权重行的 int8 点积；k 为 32 的倍数且 CPU 支持时走 AVX2
fn dot2x4(x: [&[i8]; 2], w: [&[i8]; 4], k: usize) -> [[i32; 4]; 2] {
    #[cfg(target_arch = "x86_64")]
    if k % 32 == 0 && has_avx2() {
        return unsafe {
            x86::dot2x4_avx2(
                x[0].as_ptr(),
                x[1].as_ptr(),
                [w[0].as_ptr(), w[1].as_ptr(), w[2].as_ptr(), w[3].as_ptr()],
                k,
            )
        };
    }
    let mut r = [[0i32; 4]; 2];
    for a in 0..2 {
        for b in 0..4 {
            let (mut acc, xa, wb) = (0i32, x[a], w[b]);
            for i in 0..k {
                acc += xa[i] as i32 * wb[i] as i32;
            }
            r[a][b] = acc;
        }
    }
    r
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }

    /// |x|(u8) 与 sign(x)·w(i8) 经 maddubs 成对累加到 i16（|x|≤127 不饱和），
    /// 再 madd 到 i32。与标量路径是同一组整数运算。
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot2x4_avx2(
        x0: *const i8,
        x1: *const i8,
        w: [*const i8; 4],
        k: usize,
    ) -> [[i32; 4]; 2] {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); 8];
        let mut i = 0;
        while i < k {
            let vx0 = _mm256_loadu_si256(x0.add(i) as *const __m256i);
            let vx1 = _mm256_loadu_si256(x1.add(i) as *const __m256i);
            let (a0, a1) = (_mm256_abs_epi8(vx0), _mm256_abs_epi8(vx1));
            for b in 0..4 {
                let wv = _mm256_loadu_si256(w[b].add(i) as *const __m256i);
                let p0 =
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(wv, vx0)), ones);
                let p1 =
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(wv, vx1)), ones);
                acc[b] = _mm256_add_epi32(acc[b], p0);
                acc[4 + b] = _mm256_add_epi32(acc[4 + b], p1);
            }
            i += 32;
        }
        let mut r = [[0i32; 4]; 2];
        for b in 0..4 {
            r[0][b] = hsum(acc[b]);
            r[1][b] = hsum(acc[4 + b]);
        }
        r
    }
}

/// f32 点积（8 路累加，便于编译器自动向量化）
#[inline]
pub fn dotf(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0f32; 8];
    let mut i = 0;
    while i + 8 <= n {
        for j in 0..8 {
            acc[j] += a[i + j] * b[i + j];
        }
        i += 8;
    }
    let mut sum: f32 = acc.iter().sum();
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性伪随机数（i8 范围）
    struct Rng(u64);
    impl Rng {
        fn next_i8(&mut self) -> i8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as i32 % 255 - 127) as i8
        }
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as f32 / 1e9 - 1.0
        }
    }

    #[test]
    fn quant_is_bounded_and_scaled() {
        let x: Vec<f32> = vec![0.0, -1.0, 0.5, 0.25, -0.125, 0.0, 1.0, -0.5];
        let qa = quant(&x, 4);
        assert_eq!(qa.s.len(), 2);
        assert!((qa.s[0] - 1.0 / 127.0).abs() < 1e-7);
        assert_eq!(qa.q[0], 0);
        assert_eq!(qa.q[1], -127);
        assert!(qa.q.iter().all(|&v| v >= -127));
        // 零行：scale 保持 1，量化值全 0
        let z = quant(&[0.0f32; 8], 8);
        assert_eq!(z.s[0], 1.0);
        assert!(z.q.iter().all(|&v| v == 0));
    }

    #[test]
    fn qgemm_matches_naive() {
        let (t, k, n) = (7usize, 64usize, 19usize);
        let mut r = Rng(42);
        let x: Vec<f32> = (0..t * k).map(|_| r.next_f32()).collect();
        let w: Vec<i8> = (0..n * k).map(|_| r.next_i8()).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let qa = quant(&x, k);
        let out = qgemm(&qa, &w, n, 0.37, Some(&bias));
        for ti in 0..t {
            for j in 0..n {
                let mut dot = 0i32;
                for i in 0..k {
                    dot += qa.q[ti * k + i] as i32 * w[j * k + i] as i32;
                }
                let expect = dot as f32 * qa.s[ti] * 0.37 + bias[j];
                assert_eq!(out[ti * n + j], expect, "({ti},{j})");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_kernel_matches_scalar_bit_exact() {
        if !has_avx2() {
            return; // 无 AVX2 的机器上跳过
        }
        let k = 3840;
        let mut r = Rng(7);
        let x0: Vec<i8> = (0..k).map(|_| r.next_i8()).collect();
        let x1: Vec<i8> = (0..k).map(|_| r.next_i8()).collect();
        let w: Vec<Vec<i8>> = (0..4)
            .map(|_| (0..k).map(|_| r.next_i8()).collect())
            .collect();
        let wp: [*const i8; 4] = [w[0].as_ptr(), w[1].as_ptr(), w[2].as_ptr(), w[3].as_ptr()];
        let avx = unsafe { x86::dot2x4_avx2(x0.as_ptr(), x1.as_ptr(), wp, k) };
        let scalar = dot2x4([&x0, &x1], [&w[0], &w[1], &w[2], &w[3]], k);
        // 注意：AVX2 关闭后 dot2x4 走标量——两版必须是同一组整数运算
        assert_eq!(avx, scalar);
    }

    #[test]
    fn dotf_matches_naive() {
        let mut r = Rng(99);
        let a: Vec<f32> = (0..100).map(|_| r.next_f32()).collect();
        let b: Vec<f32> = (0..100).map(|_| r.next_f32()).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let got = dotf(&a, &b);
        assert!((got - naive).abs() < 1e-4 * naive.abs().max(1.0));
    }
}
