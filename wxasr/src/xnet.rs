//! `.xnet` 模型容器。
//!
//! 容器只存放权重（不存放计算图）：全部只读算子记录堆在文件尾部，
//! 每条记录按张量名索引，含 dtype、量化 scale、形状与数据偏移。
//! 文件其余部分是任意布局的头部/填充，因此解析时从文件中部向后扫描
//! 候选位置，逐个尝试链式解析直到某个候选能恰好解析到文件尾。

use crate::error::{Error, Result};
use std::collections::HashMap;

/// f32 的 dtype 代码
pub const F32: u32 = 11;
/// int8 的 dtype 代码
pub const I8: u32 = 6;

/// 一条权重记录。数据不复制，以 `(off, len)` 引用容器内字节。
#[derive(Debug, Clone)]
pub struct Rec {
    /// dtype 代码（见 [`F32`] / [`I8`]）
    pub dtype: u32,
    /// 动态量化 scale；仅量化张量存在，取记录内 scale 数组的首个
    pub scale: Option<f32>,
    /// 形状
    pub dims: Vec<usize>,
    /// 数据起始字节偏移
    pub off: usize,
    /// 数据字节长度
    pub len: usize,
}

struct Cur<'a> {
    d: &'a [u8],
    p: usize,
}

impl Cur<'_> {
    fn u32(&mut self) -> Option<u32> {
        let b = self.d.get(self.p..self.p + 4)?;
        self.p += 4;
        Some(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let b = self.d.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(b)
    }
}

/// dtype → 元素字节大小
fn elem_size(dtype: u32) -> Option<usize> {
    match dtype {
        I8 | 2 | 3 => Some(1),
        4 | 5 => Some(2),
        1 | 7 | 8 | F32 => Some(4),
        9 | 10 | 12 => Some(8),
        _ => None,
    }
}

/// 从 `start` 起逐条解析。合法性约束：解析必须恰好到文件尾
/// （终结记录 = 零长度名 + 不超过 16 字节的全零填充），且记录总数
/// 足够多——这两条一起排除把随机字节误认成记录表的可能。
fn parse_chain(d: &[u8], start: usize) -> Option<HashMap<String, Rec>> {
    let mut c = Cur { d, p: start };
    let mut out = HashMap::new();
    loop {
        let n = c.u32()? as usize;
        if n == 0 {
            let rest = &d[c.p..];
            return (rest.len() <= 16 && rest.iter().all(|&b| b == 0) && out.len() > 100)
                .then_some(out);
        }
        if n > 256 {
            return None;
        }
        let name = std::str::from_utf8(c.take(n)?).ok()?.to_string();
        let dtype = c.u32()?;
        let es = elem_size(dtype)?;
        let quant = c.u32()?;
        for _ in 0..3 {
            c.u32()?; // 保留字段
        }
        let ns = c.u32()? as usize;
        c.u32()?; // 保留字段
        if ns > 4096 {
            return None;
        }
        let scales = c.take(ns * 4)?;
        let scale =
            (quant != 0 && ns > 0).then(|| f32::from_le_bytes(scales[..4].try_into().unwrap()));
        let nz = c.u32()? as usize;
        c.take(nz)?; // 未知名目的附带数据块
        let nd = c.u32()? as usize;
        if nd > 8 {
            return None;
        }
        let dims = (0..nd)
            .map(|_| c.u32().map(|v| v as usize))
            .collect::<Option<Vec<_>>>()?;
        let len = c.u32()? as usize;
        if len != dims.iter().product::<usize>() * es {
            return None;
        }
        let off = c.p;
        c.take(len)?;
        out.insert(
            name,
            Rec {
                dtype,
                scale,
                dims,
                off,
                len,
            },
        );
    }
}

/// 解析整个容器，返回 张量名 → 记录 的映射。
pub fn parse(d: &[u8]) -> Result<HashMap<String, Rec>> {
    if d.get(..4) != Some(b"XNET") {
        return Err(Error::Container("magic 不是 XNET".into()));
    }
    // 记录表首条记录的形态：u32 名长（值 1..=128，高 3 字节为 0），
    // 随后是全可见 ASCII 的名字。从头开始逐个候选尝试完整链式解析。
    let mut p = 8;
    while let Some(i) = d[p..]
        .windows(4)
        .position(|w| w[1..] == [0, 0, 0] && (1..=128).contains(&w[0]))
    {
        let at = p + i;
        let n = d[at] as usize;
        if let Some(name) = d.get(at + 4..at + 4 + n) {
            if name.iter().all(|b| b.is_ascii_graphic()) {
                if let Some(recs) = parse_chain(d, at) {
                    return Ok(recs);
                }
            }
        }
        p = at + 1;
    }
    Err(Error::Container("找不到权重记录表".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一条记录的字节序列
    fn rec_bytes(name: &str, dtype: u32, dims: &[usize], data: &[u8], quant: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(name.len() as u32).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&dtype.to_le_bytes());
        b.extend_from_slice(&(quant as u32).to_le_bytes());
        for _ in 0..3 {
            b.extend_from_slice(&0u32.to_le_bytes());
        }
        let ns = if quant { 1u32 } else { 0u32 };
        b.extend_from_slice(&ns.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        if quant {
            b.extend_from_slice(&0.5f32.to_le_bytes()); // scale
        }
        b.extend_from_slice(&0u32.to_le_bytes()); // nz
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &dim in dims {
            b.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(data);
        b
    }

    fn synthetic_container() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"XNET");
        d.extend_from_slice(&[0u8; 4]);
        d.extend_from_slice(&[1u8; 100]); // 头部填充，不构成合法记录表
                                          // 105 条记录（阈值 100 以上），最后一条是量化张量
        for i in 0..105 {
            let name = format!("layers.{i}.weight");
            let r = if i == 104 {
                rec_bytes(&name, I8, &[3, 4], &vec![7i8 as u8; 12], true)
            } else {
                rec_bytes(&name, F32, &[2, 2], &[0u8; 16], false)
            };
            d.extend_from_slice(&r);
        }
        d.extend_from_slice(&0u32.to_le_bytes()); // 终结：零长度名
        d.extend_from_slice(&[0u8; 8]); // 尾部填充（≤16 字节全零）
        d
    }

    #[test]
    fn parses_synthetic_container() {
        let d = synthetic_container();
        let recs = parse(&d).unwrap();
        assert_eq!(recs.len(), 105);
        let first = &recs["layers.0.weight"];
        assert_eq!(first.dtype, F32);
        assert_eq!(first.dims, vec![2, 2]);
        assert!(first.scale.is_none());
        assert_eq!(first.len, 16);
        let q = &recs["layers.104.weight"];
        assert_eq!(q.dtype, I8);
        assert_eq!(q.dims, vec![3, 4]);
        assert_eq!(q.scale, Some(0.5));
        assert_eq!(q.len, 12);
        // 偏移指向头部之后
        assert!(q.off > 108);
        // 偏移按记录顺序单调递增
        let o1 = recs["layers.1.weight"].off;
        let o2 = recs["layers.2.weight"].off;
        assert!(o1 > first.off);
        assert!(o2 > o1);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(parse(b"XNEQ\0\0\0\0").is_err());
    }

    #[test]
    fn rejects_garbage() {
        let mut d = b"XNET".to_vec();
        d.extend_from_slice(&vec![0xABu8; 4096]);
        assert!(parse(&d).is_err());
    }
}
