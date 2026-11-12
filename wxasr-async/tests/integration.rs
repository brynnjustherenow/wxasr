//! 集成测试：需要真实模型目录（环境变量 `WXASR_MODEL`，默认
//! `./model`）与 `../wxasr/tests/data` 下的 TTS 音频。
//! 缺失时对应测试跳过。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use wxasr::Model;
use wxasr_async::{start, AsrEvent, Config};

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

/// 收集事件直到 Final
async fn drain(mut rx: tokio::sync::mpsc::Receiver<AsrEvent>) -> Vec<AsrEvent> {
    let mut out = Vec::new();
    while let Some(ev) = rx.recv().await {
        let is_final = matches!(ev, AsrEvent::Final(_));
        out.push(ev);
        if is_final {
            break;
        }
    }
    out
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
    // 文本可能因尾静音以 Sentence 定稿、剩余部分进 Final，拼接后对拍
    let got: String = events
        .iter()
        .filter_map(|e| match e {
            AsrEvent::Sentence(t) | AsrEvent::Final(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    println!("拼接结果: {got}");
    assert!(!got.is_empty(), "应有定稿文本，事件: {events:?}");
    assert!(
        cer(&text, &got) < 0.1,
        "CER {:.1}%",
        cer(&text, &got) * 100.0
    );
}

#[tokio::test]
async fn silence_segments_into_sentences() {
    let Some(model) = load_model() else { return };
    // 三段语音之间插 1.6s 数字静音，默认 800ms 断句窗口应切成
    // Sentence ×2 + Final ×1（最后一段无尾静音，落在 Final）
    let gap = vec![0i16; 1600 * 16];
    let mut pcm = Vec::new();
    let clips = ["tts_short_1", "tts_short_2", "tts_short_3"];
    let mut refs = Vec::new();
    for (i, c) in clips.iter().enumerate() {
        if i > 0 {
            pcm.extend_from_slice(&gap);
        }
        pcm.extend_from_slice(&read_wav(&format!("{c}.wav")));
        refs.push(read_text(&format!("{c}.txt")));
    }

    let (handle, rx) = start(model, Config::default());
    for chunk in pcm.chunks(3200) {
        handle.push(chunk);
    }
    handle.finish();

    let events = tokio::time::timeout(Duration::from_secs(120), drain(rx))
        .await
        .expect("超时");
    let mut sentences: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AsrEvent::Sentence(t) => Some(t.clone()),
            _ => None,
        })
        .collect();
    let final_text = events
        .iter()
        .rev()
        .find_map(|e| match e {
            AsrEvent::Final(t) => Some(t.clone()),
            _ => None,
        })
        .expect("应有 Final");
    sentences.push(final_text);

    println!("切分出 {} 段：", sentences.len());
    for (s, r) in sentences.iter().zip(&refs) {
        println!("  [CER {:.1}%] {s}\n     参考: {r}", cer(r, s) * 100.0);
    }
    assert_eq!(
        sentences.len(),
        3,
        "应切成 2 句 + 1 个 Final: {sentences:?}"
    );
    for (s, r) in sentences.iter().zip(&refs) {
        assert!(cer(r, s) < 0.2, "CER {:.1}% 超阈值", cer(r, s) * 100.0);
    }
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
    let final_text = events
        .iter()
        .rev()
        .find_map(|e| match e {
            AsrEvent::Final(t) => Some(t.clone()),
            _ => None,
        })
        .expect("应有 Final");
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
/// 各自独立断句定稿——验证模型与计算池的并发复用。
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
        // 尾静音可能把文本以 Sentence 定稿，Final 只带剩余，需拼接
        let mut text = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                AsrEvent::Sentence(t) => text += &t,
                AsrEvent::Final(t) => {
                    text += &t;
                    break;
                }
                AsrEvent::Partial(_) => {}
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
