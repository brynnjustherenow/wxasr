//! GGUF 头元数据的轻量读取:只解析文件头的 KV 区,**不加载权重**,
//! 供调用方在加载几百 MB 权重之前展示/筛选模型(如目录列
//! `Stream: bool`)。
//!
//! [`gguf_meta`] 读三个键:`general.architecture`(家族名)、
//! `general.name`(可选)、`stt.capability.streaming`(模型自报能流式,
//! 即 transcribe-cpp `supports_streaming` 的同一来源,见其
//! transcribe-meta.cpp 的 `read_capability_kv`——家族默认不设置它,
//! parakeet 之外缺席即"不能")。格式按 GGUF v3 官方规格:小端,
//! 头 = magic `GGUF` + version u32 + tensor 数 u64 + KV 数 u64,
//! 每个 KV = 字符串键 + 值类型 u32 + 值;字符串 = u64 长度 + 字节,
//! bool = 1 字节(type 7),数组 = 元素类型 u32 + 个数 u64 + 元素。
//!
//! 它与 [`crate::Engine::streams`] 的分工:元数据是**文件宣称**
//! (加载前可知);`streams()` 是**运行时真值**(含设备 probe——元数据
//! 说能流但设备不支持时它为 `false`)。调用方用元数据决定是否构造
//! 伪流式后端,会话内部仍以 `streams()` 自动升降级,两者不一致也
//! 不会走错路径。

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use eyre::{Context, Result};

// GGUF v3 值类型(官方规格;全部小端)
const UINT8: u32 = 0;
const INT8: u32 = 1;
const UINT16: u32 = 2;
const INT16: u32 = 3;
const UINT32: u32 = 4;
const INT32: u32 = 5;
const FLOAT32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;
const UINT64: u32 = 10;
const INT64: u32 = 11;
const FLOAT64: u32 = 12;

/// 定长值类型的字节宽度;`None` = 变长(字符串/数组)
fn fixed_size(value_type: u32) -> Option<u64> {
    match value_type {
        UINT8 | INT8 | BOOL => Some(1),
        UINT16 | INT16 => Some(2),
        UINT32 | INT32 | FLOAT32 => Some(4),
        UINT64 | INT64 | FLOAT64 => Some(8),
        _ => None,
    }
}

/// GGUF 头能回答的模型元数据。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufMeta {
    /// `general.architecture`:模型家族名("parakeet"/"whisper"/
    /// "qwen3_asr"…)。缺失为空串(实际模型文件都有)。
    pub architecture: String,
    /// `stt.capability.streaming == 1`:文件自报支持流式识别。
    /// 这是文件宣称,设备能否真流以 [`crate::Engine::streams`]
    /// (运行时 probe)为准。
    pub streaming: bool,
    /// `general.name`:人读的模型名,可缺失。
    pub name: Option<String>,
}

/// 只读 GGUF 头,取出 [`GgufMeta`]。不触碰权重区,毫秒级返回;
/// 模型文件多大都只读头部与 KV 区。
pub fn gguf_meta(path: impl AsRef<Path>) -> Result<GgufMeta> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    gguf_meta_from(BufReader::new(file))
        .with_context(|| format!("could not read the GGUF header of {}", path.display()))
}

/// 从任何可读可寻址的流读 GGUF 头(单测直接喂内存字节)。
fn gguf_meta_from(mut reader: impl Read + Seek) -> Result<GgufMeta> {
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        eyre::bail!("not a GGUF file (magic {:02x?})", magic);
    }
    let version = read_u32(&mut reader)?;
    // v1/v2 的字符串长度是 u32,宽度不同;现存 stt 模型都是 v3
    if version != 3 {
        eyre::bail!("GGUF v{version} is not supported (only v3)");
    }
    let _tensor_count = read_u64(&mut reader)?; // 只读 KV 区,张量数跳过
    let kv_count = read_u64(&mut reader)?;

    let mut architecture = String::new();
    let mut streaming = false;
    let mut name = None;
    for _ in 0..kv_count {
        let key = read_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        match key.as_str() {
            "general.architecture" => {
                expect_type(&key, value_type, STRING)?;
                architecture = read_string(&mut reader)?;
            }
            "general.name" => {
                expect_type(&key, value_type, STRING)?;
                name = Some(read_string(&mut reader)?);
            }
            "stt.capability.streaming" => {
                expect_type(&key, value_type, BOOL)?;
                streaming = read_u32_sized(&mut reader, 1)? != 0;
            }
            _ => skip_value(&mut reader, value_type)?,
        }
    }
    Ok(GgufMeta {
        architecture,
        streaming,
        name,
    })
}

/// 读固定小端宽度的小整数(1/2/4/8 字节,零扩展到 u32;bool 用 1 字节)
fn read_u32_sized(reader: &mut impl Read, bytes: u32) -> std::io::Result<u32> {
    let mut buf = [0_u8; 8];
    reader.read_exact(&mut buf[..bytes as usize])?;
    Ok(u64::from_le_bytes(buf) as u32)
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    read_u32_sized(reader, 4)
}

fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0_u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// GGUF 字符串:u64 长度 + UTF-8 字节
fn read_string(reader: &mut impl Read) -> std::io::Result<String> {
    let len = read_u64(reader)? as usize;
    let mut bytes = vec![0_u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "a GGUF string is not UTF-8",
        )
    })
}

/// 认识的键类型不符 → 报错(而非静默当成不认识,把解析留在半途)
fn expect_type(key: &str, actual: u32, expected: u32) -> Result<()> {
    if actual != expected {
        eyre::bail!("{key} has GGUF value type {actual}, expected {expected}");
    }
    Ok(())
}

/// 跳过一个任意类型的值:定长按宽度 seek,字符串读长度再 seek,
/// 数组按元素宽度整体 seek(变长元素逐个递归)
fn skip_value(reader: &mut (impl Read + Seek), value_type: u32) -> Result<()> {
    if let Some(size) = fixed_size(value_type) {
        seek_forward(reader, size)?;
        return Ok(());
    }
    match value_type {
        STRING => {
            let len = read_u64(reader)?;
            seek_forward(reader, len)?;
        }
        ARRAY => {
            let element_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            if let Some(size) = fixed_size(element_type) {
                let bytes = count
                    .checked_mul(size)
                    .ok_or_else(|| eyre::eyre!("a GGUF array of {count} x {size}B overflows"))?;
                seek_forward(reader, bytes)?;
            } else {
                for _ in 0..count {
                    skip_value(reader, element_type)?;
                }
            }
        }
        other => eyre::bail!("unknown GGUF value type {other}"),
    }
    Ok(())
}

fn seek_forward(reader: &mut (impl Read + Seek), bytes: u64) -> Result<()> {
    i64::try_from(bytes)
        .map_err(|_| eyre::eyre!("a GGUF value spans {bytes}B, beyond i64 seek"))
        .and_then(|offset| {
            reader
                .seek(SeekFrom::Current(offset))
                .map(|_| ())
                .map_err(eyre::Report::from)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 小型 GGUF 头构造器(内存字节):三个目标键 + 若干需要跳过的
    /// 值类型,覆盖定长/字符串/定长数组/字符串数组/嵌套数组
    fn fixture() -> Vec<u8> {
        let put_u32 = |bytes: &mut Vec<u8>, v: u32| bytes.extend_from_slice(&v.to_le_bytes());
        let put_u64 = |bytes: &mut Vec<u8>, v: u64| bytes.extend_from_slice(&v.to_le_bytes());
        let put_string = |bytes: &mut Vec<u8>, s: &str| {
            put_u64(bytes, s.len() as u64);
            bytes.extend_from_slice(s.as_bytes());
        };
        let put_kv =
            |bytes: &mut Vec<u8>, key: &str, value_type: u32, value: &dyn Fn(&mut Vec<u8>)| {
                put_string(bytes, key);
                put_u32(bytes, value_type);
                value(bytes);
            };

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        put_u32(&mut out, 3); // version
        put_u64(&mut out, 0); // tensor 数(本读法不碰)
        put_u64(&mut out, 8); // KV 数
        put_kv(&mut out, "general.architecture", STRING, &|b| {
            put_string(b, "parakeet")
        });
        put_kv(&mut out, "general.name", STRING, &|b| {
            put_string(b, "Parakeet Unified EN 0.6B")
        });
        put_kv(&mut out, "stt.capability.streaming", BOOL, &|b| {
            b.push(1);
        });
        // 跳过路径的各形状:u32、f64、f32 数组、字符串数组、数组的数组
        put_kv(&mut out, "some.u32", UINT32, &|b| put_u32(b, 7));
        put_kv(&mut out, "some.f64", FLOAT64, &|b| {
            b.extend_from_slice(&1.5_f64.to_le_bytes())
        });
        put_kv(&mut out, "some.f32s", ARRAY, &|b| {
            put_u32(b, FLOAT32);
            put_u64(b, 3);
            b.extend_from_slice(
                &[0.0_f32; 3]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        });
        put_kv(&mut out, "some.strings", ARRAY, &|b| {
            put_u32(b, STRING);
            put_u64(b, 2);
            put_string(b, "a");
            put_string(b, "bb");
        });
        put_kv(&mut out, "some.nested", ARRAY, &|b| {
            put_u32(b, ARRAY);
            put_u64(b, 1);
            put_u32(b, STRING);
            put_u64(b, 1);
            put_string(b, "deep");
        });
        out
    }

    /// 写临时文件再解析(与真实用法同路径:File + BufReader + Seek)
    fn parse_fixture(bytes: &[u8]) -> Result<GgufMeta> {
        let path = std::env::temp_dir().join(format!(
            "hex-asr-gguf-meta-{}-{:x}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, bytes).expect("write the fixture");
        let meta = gguf_meta(&path);
        let _ = std::fs::remove_file(&path);
        meta
    }

    #[test]
    fn reads_the_three_keys_and_skips_everything_else() {
        let meta = parse_fixture(&fixture()).expect("parse the fixture");
        assert_eq!(meta.architecture, "parakeet");
        assert_eq!(meta.name.as_deref(), Some("Parakeet Unified EN 0.6B"));
        assert!(
            meta.streaming,
            "the fixture sets stt.capability.streaming=1"
        );
    }

    #[test]
    fn a_header_without_the_capability_reports_not_streaming() {
        // 只有 architecture 一个 KV:streaming 缺席 = false,name 缺席 = None
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        let key = "general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&STRING.to_le_bytes());
        bytes.extend_from_slice(&(7_u64).to_le_bytes());
        bytes.extend_from_slice(b"whisper");
        let meta = parse_fixture(&bytes).expect("parse the fixture");
        assert_eq!(meta.architecture, "whisper");
        assert!(!meta.streaming, "absent capability means not streaming");
        assert_eq!(meta.name, None);
    }

    #[test]
    fn a_bad_magic_is_rejected() {
        let mut bytes = fixture();
        bytes[0] = b'X';
        let error = parse_fixture(&bytes).expect_err("bad magic must fail");
        assert!(
            format!("{error:#}").contains("not a GGUF"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn a_known_key_with_the_wrong_type_is_rejected() {
        // stt.capability.streaming 写成 u32:必须报错,不能静默跳过
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        let key = "stt.capability.streaming";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&UINT32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        let error = parse_fixture(&bytes).expect_err("a wrong type must fail");
        assert!(
            format!("{error:#}").contains("stt.capability.streaming"),
            "the error names the key: {error:#}"
        );
    }
}
