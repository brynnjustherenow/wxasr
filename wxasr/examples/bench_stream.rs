//! 流式模式基准：与 bench/wtlocal-standalone 的 wtl_stream 相同逻辑——
//! 320 样本块推送、fresh>=30 节流出中间结果，计时全程。
//!
//! `cargo run --release --example bench_stream -- <模型目录> a.wav [b.wav ...]`

use std::time::Instant;
use wxasr::Model;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        std::process::exit(2);
    }
    let m = Model::load(&args[0]).expect("load model");
    for f in &args[1..] {
        let mut reader = hound::WavReader::open(f).unwrap();
        let pcm: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        let t = Instant::now();
        let mut s = m.stream();
        let mut partials = 0;
        for c in pcm.chunks(320) {
            s.push(c);
            if s.fresh() >= 30 {
                let p = s.partial();
                if !p.is_empty() {
                    partials += 1;
                }
            }
        }
        let txt = s.finish();
        let name = std::path::Path::new(f).file_stem().unwrap().to_string_lossy();
        let secs = pcm.len() as f64 / 16000.0;
        println!(
            "{name} audio {secs:.2}s {:.0}ms partials={partials}\t{txt}",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}
