//! hex 后端集成测试:需要 `--features hex` 编译、`HEXASR_MODEL` 指向
//! 真实 GGUF 模型(未设则全部静默跳过),例如:
//!
//! ```text
//! HEXASR_MODEL=path/to/model.gguf cargo test -p wxasr-async --features hex --test hex_backend
//! ```
#![cfg(feature = "hex")]

use std::sync::{Arc, OnceLock};

use wxasr_async::backend_hex::HexBackend;
use wxasr_async::{start_with, AsrEvent, AsrText, AudioIn, Config};

fn model_path() -> Option<String> {
    std::env::var("HEXASR_MODEL")
        .ok()
        .filter(|path| !path.is_empty())
}

fn device() -> hex_asr::Device {
    match std::env::var("HEXASR_DEVICE").ok().as_deref() {
        Some("gpu") => hex_asr::Device::Gpu,
        Some("cpu") => hex_asr::Device::Cpu,
        _ => hex_asr::Device::Auto,
    }
}

/// 测试 wav(wxasr 的 TTS 样本,16 kHz 单声道 i16)
fn wav_16k_mono() -> Vec<i16> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../wxasr/tests/data/tts_short_1.wav"
    );
    let mut reader = hound::WavReader::open(path).expect("打开测试 wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.channels, 1);
    reader.samples::<i16>().map(Result::unwrap).collect()
}

/// 16 kHz 单声道 → 44.1 kHz 立体声(线性插值):证明后端内部重采样/
/// 降混后与直通格式得到同样的识别结果。
fn up_to_44k1_stereo(pcm: &[i16]) -> Vec<i16> {
    let ratio = 44_100.0 / 16_000.0;
    let out_len = (pcm.len() as f64 * ratio) as usize;
    (0..out_len)
        .flat_map(|i| {
            let position = i as f64 / ratio;
            let left = position.floor() as usize;
            let frac = position - left as f64;
            let a = pcm[left] as f64;
            let b = pcm[(left + 1).min(pcm.len() - 1)] as f64;
            let sample = (a + (b - a) * frac) as i16;
            [sample, sample]
        })
        .collect()
}

/// 引擎只加载一次(模型常驻),所有会话共享。
fn engine(path: &str) -> Arc<hex_asr::Engine> {
    static ENGINE: OnceLock<Option<Arc<hex_asr::Engine>>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let spec = hex_asr::ModelSpec::new(path);
            Some(Arc::new(
                hex_asr::Engine::load(&spec, device())
                    .unwrap_or_else(|error| panic!("加载 {path} 失败: {error:#}")),
            ))
        })
        .clone()
        .expect("HEXASR_MODEL 已设置,引擎应加载成功")
}

/// 用声明的格式跑一次会话,返回唯一的 Final。
fn recognize(path: &str, audio: AudioIn, samples: &[i16]) -> AsrText {
    let (handle, mut rx) = start_with(
        Box::new(HexBackend::new(engine(path))),
        Config::default(),
        audio,
    );
    handle.push(samples);
    handle.finish();
    let mut finals = Vec::new();
    while let Some(event) = rx.blocking_recv() {
        if let AsrEvent::Final(text) = event {
            finals.push(text);
        }
    }
    assert_eq!(finals.len(), 1, "hex 会话恰一个 Final: {finals:?}");
    finals.pop().unwrap()
}

/// 同一模型、同一音频:16 kHz 单声道直通 vs 44.1 kHz 立体声(经后端
/// 内部降混/重采样),识别结果应一致——证明统一层的格式声明契约。
#[test]
fn hex_backend_recognizes_any_declared_format() {
    let Some(path) = model_path() else {
        return;
    };
    let pcm = wav_16k_mono();
    let stereo = up_to_44k1_stereo(&pcm);

    let direct = recognize(&path, AudioIn::MONO_16K, &pcm);
    let converted = recognize(
        &path,
        AudioIn {
            sample_rate: 44_100,
            channels: 2,
        },
        &stereo,
    );

    println!("16k       : {:?}", direct.content);
    println!("44.1k 立体: {:?}", converted.content);
    println!(
        "language  : {:?} / {:?}",
        direct.language, converted.language
    );
    assert!(!direct.content.trim().is_empty(), "直通格式应识别出文本");
    assert!(
        !converted.content.trim().is_empty(),
        "重采样/降混格式应识别出文本"
    );
    assert_eq!(direct.language, converted.language);
}
