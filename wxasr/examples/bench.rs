//! 基准测试：与 wtlocal 原版 wtl 的 batch 模式相同的方法学——
//! wav 读取在计时外，计时覆盖 Stream push（fbank）+ finish（定稿解码）。
//!
//! `cargo run --release --example bench -- <模型目录> a.wav [b.wav ...]`

use std::time::Instant;
use wxasr::Model;

/// 与 tests/integration.rs 相同的字错率（只比较字母数字）
fn cer(reference: &str, hypothesis: &str) -> f64 {
    let norm = |s: &str| -> Vec<char> { s.chars().filter(|c| c.is_alphanumeric()).collect() };
    let (r, h) = (norm(reference), norm(hypothesis));
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut dp: Vec<Vec<u32>> = vec![vec![0; h.len() + 1]; r.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i as u32;
    }
    for j in 0..=h.len() {
        dp[0][j] = j as u32;
    }
    for i in 1..=r.len() {
        for j in 1..=h.len() {
            let cost = (r[i - 1] != h[j - 1]) as u32;
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[r.len()][h.len()] as f64 / r.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("用法: bench <模型目录> a.wav [b.wav ...]");
        std::process::exit(2);
    }
    let t = Instant::now();
    let m = Model::load(&args[0]).expect("load model");
    eprintln!("load {:.0}ms", t.elapsed().as_secs_f64() * 1e3);
    for f in &args[1..] {
        let mut reader = hound::WavReader::open(f).unwrap();
        let pcm: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        let t = Instant::now();
        let mut s = m.stream();
        for c in pcm.chunks(320) {
            s.push(c);
        }
        let txt = s.finish();
        let cost = t.elapsed().as_secs_f64();
        let name = std::path::Path::new(f)
            .file_stem()
            .unwrap()
            .to_string_lossy();
        let sidecar = std::path::Path::new(f).with_extension("txt");
        let cer_info = std::fs::read_to_string(&sidecar)
            .ok()
            .map(|ref_text| format!("\tCER {:.1}%", cer(ref_text.trim(), &txt) * 100.0));
        println!(
            "{name} audio {:.2}s {:.0}ms\t{txt}{}",
            pcm.len() as f64 / 16000.0,
            cost * 1e3,
            cer_info.unwrap_or_default()
        );
    }
}
