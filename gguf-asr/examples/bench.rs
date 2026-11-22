//! gguf-asr bench: whole-clip offline and live-stream timing.
//!
//! cargo run -p gguf-asr --release --example bench -- <model.gguf> <wav>... [--lang zh] [--device cpu|gpu|auto] [--repeat N]
use gguf_asr::{Device, Engine, ModelSpec};
use std::time::Instant;

fn wav(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open wav");
    let spec = r.spec();
    let ch = spec.channels as usize;
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => r.samples::<i16>().map(|v| v.unwrap() as f32 / 32768.0).collect(),
        hound::SampleFormat::Float => r.samples::<f32>().map(|v| v.unwrap()).collect(),
    };
    let mono: Vec<f32> = s.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    if spec.sample_rate != gguf_asr::SAMPLE_RATE {
        gguf_asr::audio::resample(&mono, spec.sample_rate).expect("resample")
    } else {
        mono
    }
}

fn main() {
    let mut model = None;
    let mut wavs = Vec::new();
    let (mut lang, mut device, mut repeat) = (None, Device::Auto, 1usize);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lang" => lang = it.next(),
            "--device" => {
                device = match it.next().as_deref() {
                    Some("cpu") => Device::Cpu,
                    Some("gpu") => Device::Gpu,
                    _ => Device::Auto,
                }
            }
            "--repeat" => repeat = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            _ if model.is_none() => model = Some(a),
            _ => wavs.push(a),
        }
    }
    let model = model.expect("usage: bench <model.gguf> <wav>...");
    let mut spec = ModelSpec::new(&model);
    spec.language = lang;
    let t = Instant::now();
    let engine = Engine::load(&spec, device).expect("load");
    println!("load {}ms streams={}", t.elapsed().as_millis(), engine.streams());

    for path in &wavs {
        let one = wav(path);
        let mut clip = Vec::with_capacity(one.len() * repeat);
        for _ in 0..repeat {
            clip.extend_from_slice(&one);
        }
        let audio_s = clip.len() as f64 / 16000.0;
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();

        let _ = engine.transcribe_clip(clip.clone()).unwrap().wait();
        let t = Instant::now();
        let r = engine.transcribe_clip(clip.clone()).unwrap().wait().expect("offline");
        let off = t.elapsed().as_millis();
        println!("{name} audio {audio_s:.2}s offline {off}ms rtf {:.3}\t{}", off as f64 / 1000.0 / audio_s, r.text());

        if let Some(mut s) = engine.begin() {
            let t = Instant::now();
            let mut partials = 0;
            for c in clip.chunks(1600) {
                s.feed(c);
                if s.take_partial().is_some() {
                    partials += 1;
                }
            }
            let fed = t.elapsed().as_millis();
            let t2 = Instant::now();
            let r = engine.finish(Some(s), clip.clone()).unwrap().wait().expect("stream");
            println!(
                "{name} stream feed {fed}ms finish {}ms partials={partials} source={}\t{}",
                t2.elapsed().as_millis(),
                r.source.label(),
                r.text()
            );
        }
    }
}
