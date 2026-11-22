//! 声学模型：卷积前端 + 40 层 pre-norm Transformer + CTC 输出层。
//! int8 权重直接以偏移量引用 mmap 的容器数据，加载零拷贝。

use crate::error::{Error, Result};
use crate::fbank::DIM;
use crate::kernels::{dotf, qgemm, quant, QAct};
use crate::xnet::{self, Rec, F32, I8};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// —— 结构常量（由容器内张量形状交叉验证）——
const D: usize = 512; // Transformer 隐层
const HEADS: usize = 8;
const HD: usize = D / HEADS; // 64
const FF: usize = 2048; // FFN 中间层
const LAYERS: usize = 40;
const F1: usize = 32; // 前端卷积通道
const F2: usize = 64;
const F3: usize = 192;
const K2: usize = F1 * 49; // 第二层卷积核元素数
const K3: usize = F2 * 9; // 第三层卷积核元素数
const W2: usize = DIM + 1; // 第二层输出频率维（40 卷积输出 + 1 列零填充）
const FLAT: usize = F3 * 20; // 第三层输出展平长度 3840
const ATT_SCALE: f32 = 1.0 / 8.0; // = 1/sqrt(64)

/// 静态 int8 权重：容器内偏移 + 形状 + 每张量 scale
struct QW {
    off: usize,
    n: usize,
    k: usize,
    s: f32,
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

struct Layer {
    pn: Norm, // attention 前的 norm
    mn: Norm, // FFN 前的 norm
    q: QW,
    k: QW,
    v: QW,
    o: QW,
    f1: QW,
    f1b: Vec<f32>,
    f2: QW,
    f2b: Vec<f32>,
}

/// 声学模型。加载自一个目录（`.xnet` 容器 + `dict.decoder.utf8.txt` 词表）。
pub struct Model {
    map: Mmap,
    c1: Vec<f32>, // [32,1,5,5]
    c2: Vec<f32>, // [64,32,7,7]
    c3: Vec<f32>, // [192,64,3,3]
    lin: QW,
    lin_b: Vec<f32>,
    layers: Vec<Layer>,
    out: QW,
    out_b: Vec<f32>,
    vocab: Vec<String>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("layers", &self.layers.len())
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

struct Weights<'a> {
    data: &'a [u8],
    recs: HashMap<String, Rec>,
}

impl Weights<'_> {
    fn rec(&self, name: &str) -> Result<&Rec> {
        self.recs
            .get(name)
            .ok_or_else(|| Error::Model(format!("容器缺少张量 {name}")))
    }

    /// 取 f32 张量并校验形状
    fn f32(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        let r = self.rec(name)?;
        if r.dtype != F32 || r.dims != dims {
            return Err(Error::Model(format!(
                "{name} 形状不符：期望 {dims:?}，实际 {:?}",
                r.dims
            )));
        }
        Ok(self.data[r.off..r.off + r.len]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    /// 取量化权重 `{name}.weight_quantized_s8`，形状须为 [n, k]
    fn qw(&self, name: &str, n: usize, k: usize) -> Result<QW> {
        let name = format!("{name}.weight_quantized_s8");
        let r = self.rec(&name)?;
        if r.dtype != I8 || r.dims.len() < 2 || r.dims[..2] != [n, k] || r.len != n * k {
            return Err(Error::Model(format!(
                "{name} 形状不符：期望 [{n},{k}]，实际 {:?}",
                r.dims
            )));
        }
        Ok(QW {
            off: r.off,
            n,
            k,
            s: r.scale
                .ok_or_else(|| Error::Model(format!("{name} 缺 scale")))?,
        })
    }

    fn norm(&self, name: &str) -> Result<Norm> {
        Ok(Norm {
            g: self.f32(&format!("{name}.weight"), &[D])?,
            b: self.f32(&format!("{name}.bias"), &[D])?,
        })
    }
}

fn load_vocab(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| Error::Model(format!("词表不存在: {}", path.display())))?;
    let vocab: Vec<String> = text
        .lines()
        .map(|l| l.split(' ').nth(1).unwrap_or("").to_string())
        .collect();
    if vocab.is_empty() {
        return Err(Error::Model("词表为空".into()));
    }
    Ok(vocab)
}

impl Model {
    /// 从目录加载模型。目录内需有一个 `.xnet` 容器与 `dict.decoder.utf8.txt` 词表；
    /// 全部形状与词表规模校验通过后才返回。
    pub fn load(dir: impl AsRef<Path>) -> Result<Arc<Model>> {
        let dir = dir.as_ref();
        let xnet_path = std::fs::read_dir(dir)
            .map_err(|_| Error::Model(format!("模型目录不可读: {}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|e| e == "xnet"))
            .ok_or_else(|| Error::Model(format!("目录中未找到 .xnet 文件: {}", dir.display())))?;
        let map = unsafe { Mmap::map(&std::fs::File::open(&xnet_path)?)? };
        #[cfg(unix)]
        let _ = map.advise(memmap2::Advice::WillNeed);
        let w = Weights {
            data: &map,
            recs: xnet::parse(&map)?,
        };

        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("layers.{i}.");
            layers.push(Layer {
                pn: w.norm(&format!("{p}prior_norm"))?,
                mn: w.norm(&format!("{p}middle_norm"))?,
                q: w.qw(&format!("{p}attention.q_conv"), D, D)?,
                k: w.qw(&format!("{p}attention.k_conv"), D, D)?,
                v: w.qw(&format!("{p}attention.v_conv"), D, D)?,
                o: w.qw(&format!("{p}attention.output_transform_conv"), D, D)?,
                f1: w.qw(&format!("{p}feed_forward.filter_conv"), FF, D)?,
                f1b: w.f32(&format!("{p}feed_forward.filter_conv.bias"), &[FF])?,
                f2: w.qw(&format!("{p}feed_forward.output_conv"), D, FF)?,
                f2b: w.f32(&format!("{p}feed_forward.output_conv.bias"), &[D])?,
            });
        }

        let vocab_len = w
            .rec("output_conv.bias")?
            .dims
            .first()
            .copied()
            .unwrap_or(0);
        let vocab = load_vocab(&dir.join("dict.decoder.utf8.txt"))?;
        if vocab.len() != vocab_len {
            return Err(Error::Model(format!(
                "词表大小 {} 与输出层维度 {vocab_len} 不符",
                vocab.len()
            )));
        }

        Ok(Arc::new(Model {
            c1: w.f32("frontend.conv2d.weight", &[F1, 1, 5, 5])?,
            c2: w.f32("frontend.conv2d_1.weight", &[F2, F1, 7, 7])?,
            c3: w.f32("frontend.conv2d_2.weight", &[F3, F2, 3, 3])?,
            lin: w.qw("frontend.linear", D, FLAT)?,
            lin_b: w.f32("frontend.linear.bias", &[D])?,
            layers,
            out: w.qw("output_conv", vocab_len, D)?,
            out_b: w.f32("output_conv.bias", &[vocab_len])?,
            vocab,
            map,
        }))
    }

    /// 词表（CTC token → 文本片段）
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    fn mm(&self, x: &QAct, q: &QW, bias: Option<&[f32]>) -> Vec<f32> {
        let bytes = &self.map[q.off..q.off + q.n * q.k];
        // int8 数据对齐要求为 1，直接按 i8 视图使用
        let w = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) };
        qgemm(x, w, q.n, q.s, bias)
    }

    /// 整段前向：输入归一化特征帧（10ms/帧），输出每个 50ms 输出帧的
    /// argmax token id（5 帧特征 → 1 个输出帧）。
    pub fn forward(&self, x: &[[f32; DIM]]) -> Vec<u32> {
        // 独立小线程池：本负载 4 线程以上收益很小，徒增调度开销
        static POOL: std::sync::LazyLock<rayon::ThreadPool> = std::sync::LazyLock::new(|| {
            let n = std::thread::available_parallelism()
                .map_or(2, |n| n.get())
                .min(4);
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("wxasr-{i}"))
                .build()
                .expect("线程池构建失败")
        });
        POOL.install(|| self.run(x))
    }

    fn run(&self, x: &[[f32; DIM]]) -> Vec<u32> {
        let t = x.len();
        if t < 5 {
            return Vec::new();
        }
        let t1 = (t - 5) / 5 + 1;
        // 越界（负值）返回 None 的索引助手；上界由调用方各自检查
        let at = |i: usize, d: isize| -> Option<usize> { i.checked_add_signed(d) };

        // conv1 5×5, 1→32，时/频各补 2 → [t][32×40]
        let mut y1 = vec![0f32; t * F1 * DIM];
        y1.par_chunks_mut(F1 * DIM)
            .enumerate()
            .for_each(|(ti, row)| {
                let mut patch = [[0f32; 25]; DIM];
                for (f, pf) in patch.iter_mut().enumerate() {
                    for di in 0..5 {
                        let Some(tt) = at(ti, di as isize - 2).filter(|&v| v < t) else {
                            continue;
                        };
                        for dj in 0..5 {
                            if let Some(ff) = at(f, dj as isize - 2).filter(|&v| v < DIM) {
                                pf[di * 5 + dj] = x[tt][ff];
                            }
                        }
                    }
                }
                for o in 0..F1 {
                    let w = &self.c1[o * 25..o * 25 + 25];
                    for f in 0..DIM {
                        row[o * DIM + f] = dotf(w, &patch[f]).max(0.0);
                    }
                }
            });

        // conv2 7×7, 32→64，时间步长 5（时补 1、频补 3）→ [t1][64×41]
        // （频率维第 41 列恒为 0，供 conv3 取窗）
        let mut y2 = vec![0f32; t1 * F2 * W2];
        y2.par_chunks_mut(F2 * W2)
            .enumerate()
            .for_each(|(ti, row)| {
                let mut patch = vec![0f32; DIM * K2];
                for di in 0..7 {
                    let Some(tt) = at(ti * 5 + di, -1).filter(|&v| v < t) else {
                        continue;
                    };
                    let src = &y1[tt * F1 * DIM..(tt + 1) * F1 * DIM];
                    for c in 0..F1 {
                        for f in 0..DIM {
                            for dj in 0..7 {
                                let ff = f + dj; // 实际频率 = ff - 3
                                if (3..DIM + 3).contains(&ff) {
                                    patch[f * K2 + c * 49 + di * 7 + dj] = src[c * DIM + ff - 3];
                                }
                            }
                        }
                    }
                }
                for o in 0..F2 {
                    let w = &self.c2[o * K2..(o + 1) * K2];
                    for f in 0..DIM {
                        row[o * W2 + f] = dotf(w, &patch[f * K2..(f + 1) * K2]).max(0.0);
                    }
                }
            });

        // conv3 3×3, 64→192，频率步长 2（时补 1）→ [t1][192×20]
        let mut y3 = vec![0f32; t1 * FLAT];
        y3.par_chunks_mut(FLAT).enumerate().for_each(|(ti, row)| {
            let mut patch = vec![0f32; 20 * K3];
            for di in 0..3 {
                let raw = ti + di;
                if raw == 0 {
                    continue; // 对应 ti+di-1 = -1
                }
                let tt = raw - 1;
                if tt >= t1 {
                    continue;
                }
                let src = &y2[tt * F2 * W2..(tt + 1) * F2 * W2];
                for c in 0..F2 {
                    for f in 0..20 {
                        for dj in 0..3 {
                            patch[f * K3 + c * 9 + di * 3 + dj] = src[c * W2 + 2 * f + dj];
                        }
                    }
                }
            }
            for o in 0..F3 {
                let w = &self.c3[o * K3..(o + 1) * K3];
                for f in 0..20 {
                    row[o * 20 + f] = dotf(w, &patch[f * K3..(f + 1) * K3]).max(0.0);
                }
            }
        });

        // 展平 → linear 3840→512
        let mut h = self.mm(&quant(&y3, FLAT), &self.lin, Some(&self.lin_b));

        // 40 层 pre-norm Transformer
        for l in &self.layers {
            let y = quant(&layernorm(&h, &l.pn), D);
            let q = self.mm(&y, &l.q, None);
            let k = self.mm(&y, &l.k, None);
            let v = self.mm(&y, &l.v, None);
            let att = attention(&q, &k, &v, t1);
            add(&mut h, &self.mm(&quant(&att, D), &l.o, None));
            let mut f = self.mm(&quant(&layernorm(&h, &l.mn), D), &l.f1, Some(&l.f1b));
            f.par_iter_mut().for_each(|v| *v = gelu(*v));
            add(&mut h, &self.mm(&quant(&f, FF), &l.f2, Some(&l.f2b)));
        }

        // CTC 输出：逐帧 argmax
        let logits = self.mm(&quant(&h, D), &self.out, Some(&self.out_b));
        logits
            .par_chunks(self.out.n)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                        if v > b.1 {
                            (i, v)
                        } else {
                            b
                        }
                    })
                    .0 as u32
            })
            .collect()
    }

    /// CTC 折叠 + 词表映射：去连续重复、去 blank（id 0）、`<UNK>` 丢弃、
    /// `<SPACE>` → 空格、`|word` → 英文词（前一个字符是字母数字时补空格）。
    pub fn text(&self, ids: &[u32]) -> String {
        ctc_text(ids, &self.vocab)
    }
}

/// CTC 折叠与词表映射（独立出来便于单测）
fn ctc_text(ids: &[u32], vocab: &[String]) -> String {
    let mut s = String::new();
    let mut prev = u32::MAX;
    for &id in ids {
        if id != prev && id != 0 {
            match vocab.get(id as usize).map(String::as_str) {
                None | Some("<UNK>") => {}
                Some("<SPACE>") => s.push(' '),
                Some(t) if t.len() > 1 && t.starts_with('|') => {
                    if s.ends_with(|c: char| c.is_ascii_alphanumeric()) {
                        s.push(' ');
                    }
                    s.push_str(&t[1..]);
                }
                Some(t) => s.push_str(t),
            }
        }
        prev = id;
    }
    s
}

/// LayerNorm（biased 方差，eps 在根号内）
fn layernorm(x: &[f32], n: &Norm) -> Vec<f32> {
    let mut y = x.to_vec();
    y.par_chunks_mut(D).for_each(|r| {
        let mean = r.iter().sum::<f32>() / D as f32;
        let var = r.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / D as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for i in 0..D {
            r[i] = (r[i] - mean) * inv * n.g[i] + n.b[i];
        }
    });
    y
}

fn add(h: &mut [f32], o: &[f32]) {
    h.par_iter_mut()
        .zip(o.par_iter())
        .for_each(|(a, b)| *a += b);
}

/// GELU（tanh 近似）
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

/// 8 头全注意力（非因果），输出拼回 [t, D]
fn attention(q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
    // K/V 重排为按头连续 [HEADS][t][HD]:内层扫 j 时顺序访存,
    // 而非每行跨 D 步长只取 HD 个元素(原布局每个 cache line 只用 1/8)
    let heads = |x: &[f32]| {
        let mut y = vec![0f32; t * D];
        for j in 0..t {
            for h in 0..HEADS {
                y[(h * t + j) * HD..(h * t + j + 1) * HD]
                    .copy_from_slice(&x[j * D + h * HD..j * D + (h + 1) * HD]);
            }
        }
        y
    };
    let (k, v) = (heads(k), heads(v));
    let mut out = vec![0f32; t * D];
    out.par_chunks_mut(D).enumerate().for_each_init(|| vec![0f32; t], |scores, (i, o)| {
        for h in 0..HEADS {
            let qs = &q[i * D + h * HD..i * D + (h + 1) * HD];
            let kh = &k[h * t * HD..(h + 1) * t * HD];
            for (s, kr) in scores.iter_mut().zip(kh.chunks_exact(HD)) {
                *s = dotf(qs, kr) * ATT_SCALE;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                z += *s;
            }
            let oh = &mut o[h * HD..(h + 1) * HD];
            let vhs = &v[h * t * HD..(h + 1) * t * HD];
            let inv = 1.0 / z;
            for (&a, vh) in scores.iter().zip(vhs.chunks_exact(HD)) {
                let w = a * inv;
                for d in 0..HD {
                    oh[d] += w * vh[d];
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vocab() -> Vec<String> {
        // 0 blank, 1 UNK, 2 SPACE, 3 你, 4 好, 5 |hello, 6 ，
        vec![
            "<BLANK>".into(),
            "<UNK>".into(),
            "<SPACE>".into(),
            "你".into(),
            "好".into(),
            "|hello".into(),
            "，".into(),
        ]
    }

    #[test]
    fn ctc_collapse_rules() {
        let vocab = test_vocab();
        let t = |ids: &[u32]| ctc_text(ids, &vocab);
        // 去重
        assert_eq!(t(&[3, 3, 4]), "你好");
        // blank 打断的重复是两次独立发音，保留
        assert_eq!(t(&[3, 0, 3, 4]), "你你好");
        // UNK 丢弃
        assert_eq!(t(&[1, 3, 1, 4]), "你好");
        // SPACE
        assert_eq!(t(&[3, 2, 4]), "你 好");
        // 英文词：前字符非字母数字不加空格
        assert_eq!(t(&[3, 5, 4]), "你hello好");
        // 英文词：前字符是字母数字加空格
        assert_eq!(t(&[5, 5]), "hello"); // 连续重复折叠成一个
        assert_eq!(t(&[5, 4, 5]), "hello好hello");
        // 标点原样保留
        assert_eq!(t(&[3, 6, 4]), "你，好");
    }

    #[test]
    fn gelu_and_norm_basics() {
        assert_eq!(gelu(0.0), 0.0);
        let n = Norm {
            g: vec![1.0; D],
            b: vec![0.0; D],
        };
        let mut x = vec![0f32; D];
        x[0] = 3.0;
        x[1] = -3.0;
        let y = layernorm(&x, &n);
        let mean: f32 = y.iter().sum::<f32>() / D as f32;
        assert!(mean.abs() < 1e-5);
        // 有 γ/β 时正确应用
        let n2 = Norm {
            g: vec![2.0; D],
            b: vec![1.0; D],
        };
        let y2 = layernorm(&x, &n2);
        for i in 0..D {
            assert!((y2[i] - (2.0 * y[i] + 1.0)).abs() < 1e-4);
        }
    }
}
