//! 命令行转写示例：`cargo run --release --example cli -- <模型目录> <a.wav> [b.wav ...]`

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("用法: cli <模型目录> <a.wav> [b.wav ...]");
        std::process::exit(2);
    }
    let t0 = Instant::now();
    let model = wxasr::Model::load(&args[0])?;
    println!("模型加载 {:.0}ms", t0.elapsed().as_secs_f64() * 1e3);
    for f in &args[1..] {
        let t0 = Instant::now();
        let text = model.transcribe_wav(f)?;
        let reader = hound::WavReader::open(f)?;
        let secs = reader.duration() as f64 / reader.spec().sample_rate as f64;
        let cost = t0.elapsed().as_secs_f64();
        println!(
            "\n{}（{:.1}s 音频，{:.0}ms，实时率 {:.2}）\n  {text}",
            f,
            secs,
            cost * 1e3,
            cost / secs
        );
    }
    Ok(())
}
