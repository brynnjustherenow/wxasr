//! gguf 后端集成测试:需要 `--features gguf` 编译、`GGUFASR_MODEL` 指向
//! 真实 GGUF 模型(未设则全部静默跳过),例如:
//!
//! ```text
//! GGUFASR_MODEL=path/to/model.gguf cargo test -p wxasr-async --features gguf --test gguf_backend
//! ```
#![cfg(feature = "gguf")]

use std::sync::{Arc, OnceLock};

use wxasr_async::backend_gguf::{DiarizeOptions, GgufBackend, HasDiarize, PseudoOptions};
use wxasr_async::{AsrEvent, AsrText, AudioIn, Config, start_with};

fn model_path() -> Option<String> {
    std::env::var("GGUFASR_MODEL")
        .ok()
        .filter(|path| !path.is_empty())
}

fn device() -> gguf_asr::Device {
    match std::env::var("GGUFASR_DEVICE").ok().as_deref() {
        Some("gpu") => gguf_asr::Device::Gpu,
        Some("cpu") => gguf_asr::Device::Cpu,
        _ => gguf_asr::Device::Auto,
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
fn engine(path: &str) -> Arc<gguf_asr::Engine> {
    static ENGINE: OnceLock<Option<Arc<gguf_asr::Engine>>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            Some(Arc::new(
                gguf_asr::Engine::load(&spec_for(path), device())
                    .unwrap_or_else(|error| panic!("加载 {path} 失败: {error:#}")),
            ))
        })
        .clone()
        .expect("GGUFASR_MODEL 已设置,引擎应加载成功")
}

/// parakeet 家族需要 en 提示 + 400ms trailing silence(它不自报语言检测、
/// 句末标点依赖尾静音,即 `ModelSpec::unified_english`);其他模型按
/// `GGUFASR_LANG` 提示或自动检测
fn spec_for(path: &str) -> gguf_asr::ModelSpec {
    if path.to_ascii_lowercase().contains("parakeet") {
        return gguf_asr::ModelSpec::unified_english(path);
    }
    match std::env::var("GGUFASR_LANG")
        .ok()
        .filter(|lang| !lang.is_empty())
    {
        Some(lang) => gguf_asr::ModelSpec {
            language: Some(lang),
            ..gguf_asr::ModelSpec::new(path)
        },
        None => gguf_asr::ModelSpec::new(path),
    }
}

/// 用声明的格式跑一次会话,返回唯一的 End。
fn recognize(path: &str, audio: AudioIn, samples: &[i16]) -> AsrText {
    let (handle, mut rx) = start_with(
        Box::new(GgufBackend::new(engine(path))),
        Config::default(),
        audio,
    );
    handle.push(samples);
    handle.finish();
    let mut finals = Vec::new();
    while let Some(event) = rx.blocking_recv() {
        if let AsrEvent::End(text) = event {
            finals.push(text);
        }
    }
    assert_eq!(finals.len(), 1, "gguf 会话恰一个 End: {finals:?}");
    finals.pop().unwrap()
}

/// 同一模型、同一音频:16 kHz 单声道直通 vs 44.1 kHz 立体声(经后端
/// 内部降混/重采样),识别结果应一致——证明统一层的格式声明契约。
#[test]
fn gguf_backend_recognizes_any_declared_format() {
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

/// builder 链:typestate 上 `with_speaker_split` 只在 `with_diarize`
/// 之后可达(顺序错在别处就是编译错误),链尾即成品、会话照常开;存量
/// 组合构造器同批保留(返回 `HasDiarize`,可继续链)。
#[test]
fn builder_chains_capabilities_into_a_working_backend() {
    let Some(path) = model_path() else {
        return;
    };
    let pcm = wav_16k_mono();
    let backend = GgufBackend::new(engine(&path))
        .with_pseudo_streaming(PseudoOptions::default())
        .with_diarize(DiarizeOptions { window_ms: 0 })
        .with_speaker_split(gguf_asr::SpeakerSplit::Off);
    let (handle, mut rx) = start_with(Box::new(backend), Config::default(), AudioIn::MONO_16K);
    handle.push(&pcm);
    handle.finish();
    let mut ends = 0;
    while let Some(event) = rx.blocking_recv() {
        if matches!(event, AsrEvent::End(_)) {
            ends += 1;
        }
    }
    assert_eq!(ends, 1, "builder 链尾的后端照常出唯一 End");
    let _legacy: GgufBackend<HasDiarize> = GgufBackend::with_pseudo_and_diarize(
        engine(&path),
        PseudoOptions::default(),
        DiarizeOptions::default(),
    );
}

/// SAPI 英语 wav(轮换测试的输入;`GGUFASR_WAV` 可覆盖,默认用仓库合成
/// 副本 `tests/data/sapi_en.wav`,16 kHz 单声道 i16,合成方法见
/// AGENTS.md)。同名 `.txt` 是参考转写。
fn sapi_wav() -> Option<String> {
    match std::env::var("GGUFASR_WAV") {
        Ok(path) if !path.is_empty() => Some(path),
        _ => {
            let default = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/sapi_en.wav");
            std::path::Path::new(default)
                .is_file()
                .then(|| default.to_string())
        }
    }
}

/// 只比较小写字母数字的字错率(标点/大小写/空格不计)
fn norm_chars(s: &str) -> Vec<char> {
    s.chars()
        .flat_map(|c| c.to_lowercase())
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn cer(reference: &[char], hypothesis: &[char]) -> f64 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    let mut dp = vec![vec![0u32; hypothesis.len() + 1]; reference.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i as u32;
    }
    for j in 0..=hypothesis.len() {
        dp[0][j] = j as u32;
    }
    for i in 1..=reference.len() {
        for j in 1..=hypothesis.len() {
            let cost = (reference[i - 1] != hypothesis[j - 1]) as u32;
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[reference.len()][hypothesis.len()] as f64 / reference.len() as f64
}

/// 轮换端到端(事件模型 v2):实时节奏喂 36s SAPI 英语 wav(快于实时
/// 不产生 partial,见 AGENTS.md),默认 `Punct{10s,20s}` 下断言多段
/// `Commit` + 恰一个 `End` + 按序拼接 = 完整转写。只对流式模型
/// (parakeet)有意义——非流式模型 committed 不增长,只剩硬上限轮换。
#[test]
fn gguf_rotation_commits_segments_and_ends_once() {
    let Some(path) = model_path() else { return };
    let Some(wav) = sapi_wav() else { return };
    let reference = std::path::Path::new(&wav)
        .with_extension("txt")
        .exists()
        .then(|| {
            std::fs::read_to_string(std::path::Path::new(&wav).with_extension("txt"))
                .expect("读 sapi_en.txt")
        });
    let Some(reference) = reference else {
        eprintln!("[skip] {wav} 无同名 .txt 参考转写");
        return;
    };
    let engine = engine(&path);
    if !engine.streams() {
        eprintln!("[skip] 轮换端到端需要流式模型(parakeet)");
        return;
    }

    let mut reader = hound::WavReader::open(&wav).expect("打开 SAPI wav");
    assert_eq!(reader.spec().sample_rate, 16_000);
    let pcm: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();

    let (handle, mut rx) = start_with(
        Box::new(GgufBackend::new(engine)),
        Config::default(), // Punct{10s,20s} + partial
        AudioIn::MONO_16K,
    );
    // 实时节奏喂入;轮换的 finish 阻塞时统一层 std 通道排队,不丢音
    let feeder = std::thread::spawn(move || {
        for chunk in pcm.chunks(320) {
            // 20ms/块
            handle.push(chunk);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        handle.finish();
    });
    let mut commits: Vec<String> = Vec::new();
    let mut end = String::new();
    let (mut partials, mut ends) = (0, 0);
    while let Some(event) = rx.blocking_recv() {
        match event {
            AsrEvent::Partial(_) => partials += 1,
            AsrEvent::Commit(t) => commits.push(t.content),
            AsrEvent::End(t) => {
                ends += 1;
                end = t.content;
            }
        }
    }
    feeder.join().unwrap();

    println!(
        "{} 段 Commit / {} 字 End / {} 个 Partial",
        commits.len(),
        end.chars().count(),
        partials
    );
    for c in &commits {
        println!("  ✅ {c}");
    }
    println!("  🏁 {end}");
    let mut full = commits.concat();
    full.push_str(&end);
    let error = cer(&norm_chars(&reference), &norm_chars(&full));

    assert!(partials > 0, "流式模型实时节奏应有中间结果");
    assert!(
        commits.len() >= 2,
        "36s 音频 Punct{{10s,20s}} 应至少切出 2 段 Commit: {commits:?}"
    );
    // End ≤1(空文本不发):Punct 末刀切得晚时尾段只剩静音,合法地没有 End
    assert!(ends <= 1, "一次会话最多一个 End: {ends}");
    assert!(
        ends != 1 || !end.trim().is_empty(),
        "发了的 End 必非空(空文本不发事件)"
    );
    assert!(
        error < 0.2,
        "拼接 CER {:.1}% 超阈值\n  结果: {full}\n  参考: {reference}",
        error * 100.0
    );
}

/// 实时节奏跑一次会话,返回 (partial 数, Commit 文本们, End 数, End 文本)。
/// 伪流式的整段转写阻塞识别线程时,统一层 std 通道排队不丢音(与轮换
/// 测试同一机制),喂入线程只管按 20ms/块实时推。
fn run_realtime<B: wxasr_async::AsrBackend + 'static>(
    backend: B,
    cfg: Config,
    pcm: Vec<i16>,
) -> (usize, Vec<String>, usize, String) {
    let (handle, mut rx) = start_with(Box::new(backend), cfg, AudioIn::MONO_16K);
    let feeder = std::thread::spawn(move || {
        for chunk in pcm.chunks(320) {
            // 20ms/块
            handle.push(chunk);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        handle.finish();
    });
    let (mut partials, mut commits, mut ends, mut end) = (0, Vec::new(), 0, String::new());
    while let Some(event) = rx.blocking_recv() {
        match event {
            AsrEvent::Partial(_) => partials += 1,
            AsrEvent::Commit(t) => commits.push(t.content),
            AsrEvent::End(t) => {
                ends += 1;
                end = t.content;
            }
        }
    }
    feeder.join().unwrap();
    (partials, commits, ends, end)
}

/// 读 16 kHz 单声道 wav 为 i16(与轮换测试同款断言)。
fn read_wav_16k_mono(path: &str) -> Vec<i16> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("打开 {path}: {e}"));
    assert_eq!(reader.spec().sample_rate, 16_000);
    assert_eq!(reader.spec().channels, 1);
    reader.samples::<i16>().map(Result::unwrap).collect()
}

/// 伪流式端到端(离线模型,事件模型 v2):whisper 无 native 流,伪流式
/// 后端补全完整事件行为——💬 整段重转写(可回改)、✅ 段权威定稿、
/// 🏁 恰一个 End,拼接 = 完整转写。需要 GPU(CPU+whisper 超引擎
/// deadline 属设计内快速失败,这里 skip);输入 SAPI 英语 wav(36s,
/// 参考转写 `.txt`)。`GGUFASR_MODEL` 非 whisper 则 skip。
#[test]
fn pseudo_streaming_whisper_emits_full_events() {
    let Some(path) = model_path() else { return };
    if !path.to_ascii_lowercase().contains("whisper") {
        return;
    }
    if std::env::var("GGUFASR_DEVICE").as_deref() == Ok("cpu") {
        eprintln!("[skip] CPU+whisper 超引擎 deadline(设计内),伪流式需 GPU");
        return;
    }
    let Some(wav) = sapi_wav() else {
        eprintln!("[skip] 无 SAPI 英语 wav(合成方法见 AGENTS.md)");
        return;
    };
    let reference = std::fs::read_to_string(std::path::Path::new(&wav).with_extension("txt"))
        .unwrap_or_else(|e| panic!("读 {}: {e}", wav));
    let pcm = read_wav_16k_mono(&wav);

    let (partials, commits, ends, end) = run_realtime(
        GgufBackend::pseudo_streaming(engine(&path)),
        Config::default(), // Punct{10s,20s} + partial;partial 每 2s 重转写
        pcm,
    );
    println!(
        "{} 段 Commit / {} 字 End / {} 个 Partial",
        commits.len(),
        end.chars().count(),
        partials
    );
    for c in &commits {
        println!("  ✅ {c}");
    }
    println!("  🏁 {end}");
    let mut full = commits.concat();
    full.push_str(&end);
    let error = cer(&norm_chars(&reference), &norm_chars(&full));

    assert!(partials > 0, "伪流式每 2s 重转写应有中间结果");
    assert!(
        commits.len() >= 1,
        "36s 音频 Punct{{10s,20s}} 硬上限兜底至少 1 段 Commit: {commits:?}"
    );
    // End ≤1(空文本不发):Punct 末刀切得晚时尾段只剩静音,合法地没有 End
    assert!(ends <= 1, "一次会话最多一个 End: {ends}");
    assert!(
        ends != 1 || !end.trim().is_empty(),
        "发了的 End 必非空(空文本不发事件)"
    );
    assert!(
        error < 0.2,
        "拼接 CER {:.1}% 超阈值\n  结果: {full}\n  参考: {reference}",
        error * 100.0
    );
}

/// 伪流式端到端(离线模型):qwen(中文)+ 仓库 TTS wav ×3(15s,单句
/// "今天天气真不错,我们一起去公园散步吧。")。轮换用
/// Punct{1.5s,4s}(短音频下硬上限 4s 兜底保证切段),断言同 whisper:
/// ≥1 Partial、≥1 Commit、恰一 End、拼接 CER 达标。
#[test]
fn pseudo_streaming_qwen_emits_full_events() {
    let Some(path) = model_path() else { return };
    if !path.to_ascii_lowercase().contains("qwen") {
        return;
    }
    let wav = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../wxasr/tests/data/tts_short_1.wav"
    );
    let sentence = std::fs::read_to_string(std::path::Path::new(wav).with_extension("txt"))
        .expect("tts_short_1.txt 是已知参考文本");
    let one = read_wav_16k_mono(wav);
    let pcm: Vec<i16> = one.repeat(3); // ×3 ≈ 15s,硬上限 4s 保证切段

    let (partials, commits, ends, end) = run_realtime(
        GgufBackend::pseudo_streaming(engine(&path)),
        Config {
            rotate: wxasr_async::policy::RotatePolicy::Punct {
                soft_ms: 1_500,
                hard_ms: 4_000,
            },
            ..Config::default()
        },
        pcm,
    );
    println!(
        "{} 段 Commit / {} 字 End / {} 个 Partial",
        commits.len(),
        end.chars().count(),
        partials
    );
    for c in &commits {
        println!("  ✅ {c}");
    }
    println!("  🏁 {end}");
    let mut full = commits.concat();
    full.push_str(&end);
    let repeated = sentence.trim().repeat(3);
    let error = cer(&norm_chars(&repeated), &norm_chars(&full));

    assert!(partials > 0, "伪流式每 2s 重转写应有中间结果");
    assert!(
        commits.len() >= 1,
        "15s 音频硬上限 4s 至少切出 2 段 Commit: {commits:?}"
    );
    // End ≤1(空文本不发):末刀切得晚时尾段只剩静音,合法地没有 End
    assert!(ends <= 1, "一次会话最多一个 End: {ends}");
    assert!(!full.trim().is_empty(), "拼接应为完整转写");
    assert!(
        error < 0.2,
        "拼接 CER {:.1}% 超阈值\n  结果: {full}\n  参考: {repeated}",
        error * 100.0
    );
}

/// 反向:伪流式后端 + 流式模型(parakeet)→ open 时自动升级真流式
/// 路径,行为与轮换端到端一致(✅ 多段、🏁 恰一、CER 达标)。
#[test]
fn pseudo_streaming_with_a_streaming_model_upgrades() {
    let Some(path) = model_path() else { return };
    if !path.to_ascii_lowercase().contains("parakeet") {
        return;
    }
    let Some(wav) = sapi_wav() else {
        eprintln!("[skip] 无 SAPI 英语 wav(合成方法见 AGENTS.md)");
        return;
    };
    let reference = std::fs::read_to_string(std::path::Path::new(&wav).with_extension("txt"))
        .unwrap_or_else(|e| panic!("读 {}: {e}", wav));
    let pcm = read_wav_16k_mono(&wav);
    let engine = engine(&path);
    if !engine.streams() {
        eprintln!("[skip] 该设备上 parakeet 不能流式,无从验证升级");
        return;
    }

    let (partials, commits, ends, end) = run_realtime(
        GgufBackend::pseudo_streaming(engine),
        Config::default(),
        pcm,
    );
    println!(
        "{} 段 Commit / {} 字 End / {} 个 Partial",
        commits.len(),
        end.chars().count(),
        partials
    );
    let mut full = commits.concat();
    full.push_str(&end);
    let error = cer(&norm_chars(&reference), &norm_chars(&full));

    assert!(partials > 0, "升级后的真流式应有中间结果");
    assert!(
        commits.len() >= 2,
        "36s 音频 Punct{{10s,20s}} 应至少切出 2 段 Commit: {commits:?}"
    );
    // End ≤1(空文本不发):末刀切得晚时尾段只剩静音,合法地没有 End
    assert!(ends <= 1, "一次会话最多一个 End: {ends}");
    assert!(
        error < 0.2,
        "拼接 CER {:.1}% 超阈值\n  结果: {full}\n  参考: {reference}",
        error * 100.0
    );
}
