//! 最简实时桌面听写 CLI：采集桌面音频（系统混音），每断一句 println 一行。
//!
//! `cargo run --release --example live_desktop -- [模型目录]`
//!
//! Ctrl+C 退出。播放任意声音（音乐/视频/语音）即可看到逐句输出。

use std::sync::Arc;
use wxasr::Model;
use wxasr_async::{start, AsrEvent, Config};
use wxasr_provider::{AudioSource, DesktopAudio};

#[tokio::main]
async fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./model".to_string());
    let model = Model::load(&model_dir).expect("模型加载失败");
    let (asr, mut rx) = start(
        model,
        Config {
            silence_ms: 300,
            ..Default::default()
        },
    );
    let sink = {
        let asr = asr.clone();
        let sink: wxasr_provider::PcmSink = Arc::new(move |pcm: &[i16]| asr.push(pcm));
        sink
    };

    let source = DesktopAudio::new();
    let mut capture = source.start(None, sink).expect("桌面采集启动失败");
    println!("实时桌面听写中（播放任意声音即可）… Ctrl+C 退出");

    // 最简 loop：断句触发 println
    while let Some(ev) = rx.recv().await {
        match ev {
            AsrEvent::Sentence(t) => println!("{t}"),
            AsrEvent::Final(t) => {
                println!("{t}");
                break;
            }
            AsrEvent::Partial(_) => {}
        }
    }
    capture.stop();
}
