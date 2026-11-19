//! 诊断探针(临时):绕过引擎 job deadline,直接 `LocalTranscriber`
//! 转写,区分"引擎层卡住"与"native 推理本身不返回"。
//! 三重门控:`GGUFASR_NATIVE_PROBE=1` + `GGUFASR_MODEL` + `GGUFASR_WAV`。

use std::time::Instant;

use gguf_asr::{Device, LocalTranscriber, ModelSpec};

#[test]
fn native_transcribe_without_engine_deadline() {
    let path = std::env::var("GGUFASR_MODEL").unwrap_or_default();
    let wav = std::env::var("GGUFASR_WAV").unwrap_or_default();
    if std::env::var("GGUFASR_NATIVE_PROBE").ok().as_deref() != Some("1")
        || path.is_empty()
        || wav.is_empty()
    {
        return;
    }
    // native/ggml 诊断放行到 stderr(transcribe_cpp::init_logging 接进
    // `log` 门面,默认无人订阅则全被吞)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    let spec = if path.to_ascii_lowercase().contains("parakeet") {
        ModelSpec::unified_english(&path)
    } else {
        ModelSpec::new(&path)
    };
    let started = Instant::now();
    let device = match std::env::var("GGUFASR_DEVICE").ok().as_deref() {
        Some("cpu") => Device::Cpu,
        Some("gpu") => Device::Gpu,
        _ => Device::Gpu,
    };
    let mut transcriber =
        LocalTranscriber::load(&spec, device).unwrap_or_else(|e| panic!("load: {e:#}"));
    println!("load: {:?}", started.elapsed());
    eprintln!("STEP-0 device_label: {}", transcriber.device_label());

    let mut reader = hound::WavReader::open(&wav).expect("wav");
    assert_eq!(reader.spec().sample_rate, 16_000);
    // 整段(上限 30s,whisper 单窗),GGUFASR_PROBE_SECONDS 可截短
    let limit_ms: usize = std::env::var("GGUFASR_PROBE_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .take(16_000 * limit_ms)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let mut padded = samples;
    gguf_asr::pad_gguf_samples(&mut padded, transcriber.trailing_silence_samples());

    let t = Instant::now();
    let transcript = transcriber.transcribe(&padded).expect("transcribe");
    println!("transcribe clip: {:?}", t.elapsed());
    println!("text: {:?}", transcript.text);
    println!("language: {:?}", transcript.language);
    for (i, segment) in transcript.segments.iter().enumerate() {
        println!(
            "seg[{i}] {}-{}ms {:?}",
            segment.start_ms, segment.end_ms, segment.text
        );
    }
}

/// 最终隔离:绕过 gguf-asr 全部包装,直接调 transcribe_cpp 原始绑定,
/// 同一进程/二进制/环境。load(native 自带 warm)→ session#1 run →
/// session#2 run,复刻轮换模式。三段都有 stderr 标记。
/// `GGUFASR_RAW_PROBE=1` + `GGUFASR_MODEL` + `GGUFASR_WAV`。
#[test]
fn raw_bindings_without_gguf_asr_wrapper() {
    let path = std::env::var("GGUFASR_MODEL").unwrap_or_default();
    let wav = std::env::var("GGUFASR_WAV").unwrap_or_default();
    if std::env::var("GGUFASR_RAW_PROBE").ok().as_deref() != Some("1")
        || path.is_empty()
        || wav.is_empty()
    {
        return;
    }
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    transcribe_cpp::init_logging();
    transcribe_cpp::init_backends_default().expect("init backends");
    drop(transcribe_cpp::devices());

    let started = Instant::now();
    let model = transcribe_cpp::Model::load_with(
        &path,
        &transcribe_cpp::ModelOptions {
            backend: transcribe_cpp::Backend::Auto,
            device: None,
            ..Default::default()
        },
    )
    .expect("raw load");
    eprintln!(
        "RAW-0 load {:?} device={:?}",
        started.elapsed(),
        model.device()
    );

    let mut reader = hound::WavReader::open(&wav).expect("wav");
    let speech: Vec<f32> = reader
        .samples::<i16>()
        .take(16_000 * 3)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let mut session = model
        .session_with(&transcribe_cpp::SessionOptions::default())
        .expect("raw session 1");
    let t = Instant::now();
    let a = session
        .run(&speech, &transcribe_cpp::RunOptions::default())
        .expect("raw run 1");
    eprintln!(
        "RAW-1 first-session run {:?} text={:?}",
        t.elapsed(),
        a.text
    );

    let mut session2 = model
        .session_with(&transcribe_cpp::SessionOptions::default())
        .expect("raw session 2");
    let t = Instant::now();
    let b = session2
        .run(&speech, &transcribe_cpp::RunOptions::default())
        .expect("raw run 2");
    eprintln!(
        "RAW-2 second-session run {:?} text={:?}",
        t.elapsed(),
        b.text
    );
}

/// 差异归因:在"能跑通的原始路径"上逐个加回包装层差异,锁定挂死因子。
/// `GGUFASR_RAW_BISECT=threads` → SessionOptions{n_threads:1}(Hex 在 GPU
/// 设备上设的);`timestamps` → RunOptions{timestamps:Segment}。
/// 每个阶段先 warm 1.5s 静音(session#1)再用 session#2 跑 3s 语音,
/// 与包装层的轮换模式一致。
#[test]
fn raw_bindings_bisect_wrapper_differences() {
    let path = std::env::var("GGUFASR_MODEL").unwrap_or_default();
    let wav = std::env::var("GGUFASR_WAV").unwrap_or_default();
    let mode = std::env::var("GGUFASR_RAW_BISECT").unwrap_or_default();
    if mode.is_empty() || path.is_empty() || wav.is_empty() {
        return;
    }
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    transcribe_cpp::init_logging();
    transcribe_cpp::init_backends_default().expect("init backends");
    drop(transcribe_cpp::devices());

    let session_options = match mode.as_str() {
        "threads" | "full" => transcribe_cpp::SessionOptions {
            n_threads: 1,
            ..Default::default()
        },
        _ => transcribe_cpp::SessionOptions::default(),
    };
    let run_options = match mode.as_str() {
        "timestamps" | "full" => transcribe_cpp::RunOptions {
            timestamps: transcribe_cpp::TimestampKind::Segment,
            ..Default::default()
        },
        _ => transcribe_cpp::RunOptions::default(),
    };
    // "caps":把包装层特有的 capabilities() 调用(3 次)撒进同形状流程,
    // 验证它是否是 wedge 触发因子。
    let sprinkle_caps = mode == "caps";

    let model = transcribe_cpp::Model::load_with(
        &path,
        &transcribe_cpp::ModelOptions {
            backend: transcribe_cpp::Backend::Auto,
            device: None,
            ..Default::default()
        },
    )
    .expect("load");
    eprintln!("BISECT-{} mode={mode} loaded", std::process::id());
    if sprinkle_caps {
        let _ = model.capabilities();
    }

    let mut warm = model.session_with(&session_options).expect("warm session");
    let silence = vec![0.0_f32; 24_000];
    let t = Instant::now();
    let a = warm.run(&silence, &run_options).expect("warm run");
    eprintln!("BISECT-1 warm(silence) {:?} text={:?}", t.elapsed(), a.text);
    if sprinkle_caps {
        let _ = model.capabilities();
    }
    if mode == "doublewarm" || mode == "full" {
        // 包装层形态:load 里 warm 两次(真实转写在第 3 个 session 上)
        let mut warm2 = model.session_with(&session_options).expect("warm2 session");
        let t = Instant::now();
        let a2 = warm2.run(&silence, &run_options).expect("warm2 run");
        eprintln!(
            "BISECT-1b warm#2(silence) {:?} text={:?}",
            t.elapsed(),
            a2.text
        );
    }

    let mut reader = hound::WavReader::open(&wav).expect("wav");
    let speech: Vec<f32> = reader
        .samples::<i16>()
        .take(16_000 * 3)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let mut session = model.session_with(&session_options).expect("run session");
    if sprinkle_caps {
        let _ = model.capabilities();
    }
    let t = Instant::now();
    let b = session.run(&speech, &run_options).expect("speech run");
    eprintln!(
        "BISECT-2 speech(session#2) {:?} text={:?}",
        t.elapsed(),
        b.text
    );
}

/// 分界实验:同进程先后转写 1.5s 静音(warm 等价)与 3s 语音,
/// 区分"whisper 全坏"与"帧数/内容阈值以上才挂"。GGUFASR_PROBE_STEPS=2。
#[test]
fn native_whisper_silence_then_speech() {
    let path = std::env::var("GGUFASR_MODEL").unwrap_or_default();
    let wav = std::env::var("GGUFASR_WAV").unwrap_or_default();
    if std::env::var("GGUFASR_NATIVE_PROBE").ok().as_deref() != Some("1")
        || std::env::var("GGUFASR_PROBE_STEPS").ok().as_deref() != Some("2")
        || path.is_empty()
        || wav.is_empty()
    {
        return;
    }
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    // GGUFASR_LANG 给语言提示:跳过 whisper 的语言检测 prefill(排查该路径)
    let spec = match std::env::var("GGUFASR_LANG").ok().filter(|l| !l.is_empty()) {
        Some(lang) => ModelSpec {
            language: Some(lang),
            ..ModelSpec::new(&path)
        },
        None => ModelSpec::new(&path),
    };
    let mut transcriber =
        LocalTranscriber::load(&spec, Device::Gpu).unwrap_or_else(|e| panic!("load: {e:#}"));
    eprintln!("STEP-0 device_label: {}", transcriber.device_label());

    // 第一段:1.5s 数字静音 + trailing pad(warm_on_silence 同款输入)
    let mut silence = vec![0.0_f32; 24_000];
    gguf_asr::pad_gguf_samples(&mut silence, transcriber.trailing_silence_samples());
    let t = Instant::now();
    let a = transcriber
        .transcribe(&silence)
        .expect("silence transcribe");
    eprintln!("STEP-1 1.5s silence: {:?} text={:?}", t.elapsed(), a.text);

    // 第二段:3s 真语音 + trailing pad
    let mut reader = hound::WavReader::open(&wav).expect("wav");
    let speech: Vec<f32> = reader
        .samples::<i16>()
        .take(16_000 * 3)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let mut padded = speech;
    gguf_asr::pad_gguf_samples(&mut padded, transcriber.trailing_silence_samples());
    let t = Instant::now();
    let b = transcriber.transcribe(&padded).expect("speech transcribe");
    eprintln!("STEP-2 3s speech: {:?} text={:?}", t.elapsed(), b.text);
}
