//! 集成测试：需要真实模型目录（环境变量 `WXASR_MODEL`，默认
//! `./model`）与 `../wxasr/tests/data` 下的 TTS 音频。
//! 缺失时对应测试跳过。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use wxasr::Model;
use wxasr_async::{policy::RotatePolicy, start, AsrEvent, Config};

fn load_model() -> Option<Arc<Model>> {
    let dir = std::env::var("WXASR_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./model"));
    match Model::load(&dir) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("[skip] 模型加载失败：{e}");
            None
        }
    }
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../wxasr/tests/data")
}

fn read_wav(name: &str) -> Vec<i16> {
    let mut r = hound::WavReader::open(data_dir().join(name)).unwrap();
    r.samples::<i16>().map(|s| s.unwrap()).collect()
}

fn read_text(name: &str) -> String {
    std::fs::read_to_string(data_dir().join(name))
        .unwrap()
        .trim()
        .to_string()
}

/// 字错率（只比较字母数字字符）
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
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j as u32;
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

/// 收集事件直到通道关闭（会话结束）
async fn drain(mut rx: tokio::sync::mpsc::Receiver<AsrEvent>) -> Vec<AsrEvent> {
    let mut out = Vec::new();
    while let Some(ev) = rx.recv().await {
        out.push(ev);
    }
    out
}

/// 按顺序拼接所有 Final，得到完整转写
fn transcript(events: &[AsrEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            AsrEvent::Final(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn fast_feed_yields_final() {
    let Some(model) = load_model() else { return };
    let pcm = read_wav("tts_short_1.wav");
    let text = read_text("tts_short_1.txt");

    let (handle, rx) = start(
        model,
        Config {
            partial: false,
            ..Default::default()
        },
    );
    for chunk in pcm.chunks(3200) {
        handle.push(chunk);
    }
    handle.finish();

    let events = tokio::time::timeout(Duration::from_secs(60), drain(rx))
        .await
        .expect("超时");
    let got = transcript(&events);
    println!("定稿结果: {got}");
    assert!(!got.is_empty(), "应有定稿文本，事件: {events:?}");
    assert!(
        cer(&text, &got) < 0.1,
        "CER {:.1}%",
        cer(&text, &got) * 100.0
    );
}

/// 三段语音（每段 >5s）间插 1.6s 静音：软上限 3s 意味着第 1、2 段后的
/// 停顿必然触发静音轮换，各产生一个 Final（第 3 段由 finish 定稿）。
/// 拼接所有 Final 与三段参考对拍，验证轮换定稿不丢字、拼接无缝。
#[tokio::test]
async fn rotation_keeps_transcript_intact() {
    let Some(model) = load_model() else { return };
    let gap = vec![0i16; 1600 * 16];
    let mut pcm = Vec::new();
    let clips = ["tts_short_1", "tts_short_2", "tts_short_3"];
    let mut refs = String::new();
    for (i, c) in clips.iter().enumerate() {
        if i > 0 {
            pcm.extend_from_slice(&gap);
        }
        pcm.extend_from_slice(&read_wav(&format!("{c}.wav")));
        refs.push_str(&read_text(&format!("{c}.txt")));
    }

    let (handle, rx) = start(
        model,
        Config {
            rotate: RotatePolicy::Auto {
                soft_ms: 3000,
                hard_ms: 60_000,
            },
            ..Default::default()
        },
    );
    for chunk in pcm.chunks(3200) {
        handle.push(chunk);
    }
    handle.finish();

    let events = tokio::time::timeout(Duration::from_secs(120), drain(rx))
        .await
        .expect("超时");
    let finals: Vec<&String> = events
        .iter()
        .filter_map(|e| match e {
            AsrEvent::Final(t) => Some(t),
            _ => None,
        })
        .collect();
    println!("切分出 {} 段：", finals.len());
    for f in &finals {
        println!("  {f}");
    }
    let got = transcript(&events);

    println!("轮换拼接结果: {got}\n参考: {refs}");
    assert!(
        finals.len() >= 3,
        "两次静音轮换 + 结束定稿应至少产生 3 个 Final: {finals:?}"
    );
    assert!(
        cer(&refs, &got) < 0.2,
        "CER {:.1}% 超阈值",
        cer(&refs, &got) * 100.0
    );
}

/// 说话中硬切：三段首尾相接（无静音可借）+ 硬上限 3s 强制轮换。接续流
/// 携带左文与未定稿尾部、切点对齐字间停顿，跨切点不应丢字。
#[tokio::test]
async fn hard_rotation_mid_speech_stays_accurate() {
    let Some(model) = load_model() else { return };
    let mut pcm = Vec::new();
    let mut refs = String::new();
    for c in ["tts_short_1", "tts_short_2", "tts_short_3"] {
        pcm.extend_from_slice(&read_wav(&format!("{c}.wav")));
        refs.push_str(&read_text(&format!("{c}.txt")));
    }

    let (handle, rx) = start(
        model,
        Config {
            rotate: RotatePolicy::Auto {
                soft_ms: 60_000,
                hard_ms: 3_000,
            },
            ..Default::default()
        },
    );
    for chunk in pcm.chunks(3200) {
        handle.push(chunk);
    }
    handle.finish();

    let events = tokio::time::timeout(Duration::from_secs(120), drain(rx))
        .await
        .expect("超时");
    let finals = events
        .iter()
        .filter(|e| matches!(e, AsrEvent::Final(_)))
        .count();
    let got = transcript(&events);
    println!(
        "硬切 {finals} 段，CER {:.1}%\n  结果: {got}\n  参考: {refs}",
        cer(&refs, &got) * 100.0
    );
    assert!(
        finals >= 4,
        "18s 音频按 3s 硬切应产生多个 Final，实际 {finals}"
    );
    assert!(
        cer(&refs, &got) < 0.1,
        "CER {:.1}% 超阈值",
        cer(&refs, &got) * 100.0
    );
}

/// 标点触发轮换：软上限 1s 后遇定稿标点即切。除最后一段（finish 定稿）
/// 外，每个 Final 都应以标点收尾——行读起来是完整的句/分句；拼接后
/// 不丢字。
#[tokio::test]
async fn punct_policy_finals_end_with_punctuation() {
    const PUNCT: &[char] = &['，', '。', '？', '！', '、', ',', '.', '?', '!'];
    let Some(model) = load_model() else { return };
    let gap = vec![0i16; 1600 * 16];
    let mut pcm = Vec::new();
    let mut refs = String::new();
    for (i, c) in ["tts_short_1", "tts_short_2", "tts_short_3"]
        .iter()
        .enumerate()
    {
        if i > 0 {
            pcm.extend_from_slice(&gap);
        }
        pcm.extend_from_slice(&read_wav(&format!("{c}.wav")));
        refs.push_str(&read_text(&format!("{c}.txt")));
    }

    let (handle, rx) = start(
        model,
        Config {
            rotate: RotatePolicy::Punct {
                soft_ms: 1000,
                hard_ms: 60_000,
            },
            ..Default::default()
        },
    );
    for chunk in pcm.chunks(3200) {
        handle.push(chunk);
    }
    handle.finish();

    let events = tokio::time::timeout(Duration::from_secs(120), drain(rx))
        .await
        .expect("超时");
    let finals: Vec<&String> = events
        .iter()
        .filter_map(|e| match e {
            AsrEvent::Final(t) => Some(t),
            _ => None,
        })
        .collect();
    println!("标点切分出 {} 段：", finals.len());
    for f in &finals {
        println!("  {f}");
    }
    let got = transcript(&events);
    println!("CER {:.1}%", cer(&refs, &got) * 100.0);

    assert!(finals.len() >= 3, "应按标点切出多段: {finals:?}");
    for f in &finals[..finals.len() - 1] {
        assert!(f.ends_with(PUNCT), "标点触发的 Final 应以标点收尾: {f:?}");
    }
    assert!(
        cer(&refs, &got) < 0.15,
        "CER {:.1}% 超阈值",
        cer(&refs, &got) * 100.0
    );
}

#[tokio::test]
async fn realtime_feed_produces_partials() {
    let Some(model) = load_model() else { return };
    let pcm = read_wav("tts_short_1.wav");
    let text = read_text("tts_short_1.txt");

    let (handle, rx) = start(model, Config::default());
    let feeder = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(20));
        for chunk in pcm.chunks(320) {
            interval.tick().await;
            handle.push(chunk);
        }
        handle.finish();
    });

    let events = tokio::time::timeout(Duration::from_secs(60), drain(rx))
        .await
        .expect("超时");
    feeder.await.unwrap();

    let partials: Vec<&String> = events
        .iter()
        .filter_map(|e| match e {
            AsrEvent::Partial(t) => Some(t),
            _ => None,
        })
        .collect();
    let final_text = transcript(&events);
    println!(
        "{} 个中间结果，首个: {}",
        partials.len(),
        partials.first().map(|s| s.as_str()).unwrap_or("")
    );
    println!("Final: {final_text}");
    assert!(!partials.is_empty(), "实时节奏下应有中间结果");
    assert!(cer(&text, &final_text) < 0.1);
}

/// 双流并发：桌面 + 麦克风场景。共享同一 Arc<Model>，两路同时喂入，
/// 各自独立轮换定稿——验证模型与计算池的并发复用。
#[tokio::test]
async fn two_streams_share_one_model() {
    let Some(model) = load_model() else { return };
    let pcm_a = read_wav("tts_short_1.wav");
    let pcm_b = read_wav("tts_short_3.wav");
    let (text_a, text_b) = (read_text("tts_short_1.txt"), read_text("tts_short_3.txt"));

    let run = |pcm: Vec<i16>, thread: &'static str, model: Arc<Model>| async move {
        let (handle, mut rx) = start(
            model,
            Config {
                thread_name: Some(thread.into()),
                ..Default::default()
            },
        );
        let feeder = tokio::spawn(async move {
            for chunk in pcm.chunks(3200) {
                handle.push(chunk);
                tokio::time::sleep(Duration::from_millis(50)).await; // 模拟实时节奏
            }
            handle.finish();
        });
        // 拼接所有 Final，直到通道关闭
        let mut text = String::new();
        while let Some(ev) = rx.recv().await {
            if let AsrEvent::Final(t) = ev {
                text += &t;
            }
        }
        feeder.await.unwrap();
        text
    };

    let (a, b) = tokio::join!(
        run(pcm_a, "wxasr-desktop", model.clone()),
        run(pcm_b, "wxasr-mic", model.clone()),
    );
    println!("流A（desktop）: {a}\n流B（mic）:    {b}");
    println!(
        "CER: A {:.1}% / B {:.1}%",
        cer(&text_a, &a) * 100.0,
        cer(&text_b, &b) * 100.0
    );
    assert!(cer(&text_a, &a) < 0.1, "流A CER 超阈值");
    assert!(cer(&text_b, &b) < 0.1, "流B CER 超阈值");
}
