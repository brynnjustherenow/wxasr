//! 最简实时桌面听写 CLI：采集桌面音频（系统混音），备用屏幕滚动显示。
//!
//! `cargo run --release --example live_desktop -- [模型目录]`
//!
//! - 中间结果（Partial）：💬 当前行实时刷新
//! - 定稿（Commit/End）：✅ 换行提交，之前行不再变化（上下文轮换或结束时各一次）
//! - Ctrl+C：优雅结束——定稿当前段（✅）后退出，各行落回主屏幕；
//!   再按一次 Ctrl+C 立即强制退出
//!
//! 播放任意声音（音乐/视频/语音）即可看到实时转写。

use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::execute;
use crossterm::terminal::{size, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen};
use std::io::{self, Write};
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
    let (asr, mut rx) = start(model, Config::default());
    let sink = {
        let asr = asr.clone();
        let sink: wxasr_provider::PcmSink = Arc::new(move |pcm: &[i16]| asr.push(pcm));
        sink
    };
    let mut asr_fin = Some(asr);

    let source = DesktopAudio::new();
    let mut capture = source.start(None, sink).expect("桌面采集启动失败");

    let mut out = io::stdout();
    let _ = execute!(out, EnterAlternateScreen, Hide);

    let mut committed: Vec<String> = Vec::new(); // 定稿各行（Commit/End）
    let mut live = String::new(); // 当前段中间结果
    let mut finishing = false;
    loop {
        render(&mut out, &committed, &live);
        tokio::select! {
            ev = rx.recv() => match ev {
                Some(AsrEvent::Partial(t)) => live = t.content,
                Some(AsrEvent::Commit(t)) | Some(AsrEvent::End(t)) => {
                    committed.push(t.content);
                    live.clear();
                }
                None => break, // 通道关闭：会话结束
            },
            _ = tokio::signal::ctrl_c() => {
                if finishing {
                    break; // 第二次 Ctrl+C 强制退出
                }
                finishing = true;
                capture.stop();
                if let Some(a) = asr_fin.take() {
                    let _ = a.finish(); // 定稿当前段，等通道关闭
                }
            }
        }
    }
    let _ = execute!(out, LeaveAlternateScreen, Show);
    capture.stop();
    for l in &committed {
        println!("{l}");
    }
    if !live.is_empty() {
        println!("{live}"); // 强制退出时未定稿的部分
    }
}

/// 回到左上角、清掉以下所有行，逐行重画：✅ 定稿行在前、💬 当前行在末尾。
/// 每行按 CJK 双列保守折算截断到不折行，整体只保留最后一屏，避免滚动残影
fn render(out: &mut impl Write, committed: &[String], live: &str) {
    let (cols, rows) = size().unwrap_or((80, 24));
    let width = cols as usize / 2 - 2;
    let mut lines: Vec<String> = committed.iter().map(|l| fmt_line("✅", l, width)).collect();
    if !live.is_empty() {
        lines.push(fmt_line("💬", live, width));
    }
    if lines.is_empty() {
        lines.push(String::from(
            "🎙 实时桌面听写中（播放任意声音即可）… Ctrl+C 退出",
        ));
    }
    let lines = &lines[lines.len().saturating_sub(rows as usize)..];
    let _ = execute!(out, MoveTo(0, 0), Clear(ClearType::FromCursorDown));
    for l in lines {
        let _ = writeln!(out, "{l}");
    }
    let _ = out.flush();
}

/// icon + 单行文本，超宽时只显示尾部（前缀 …），保证物理上不折行
fn fmt_line(icon: &str, text: &str, width: usize) -> String {
    let n = text.chars().count();
    if n > width {
        let mut s = format!("{icon} …");
        s.extend(text.chars().skip(n - (width - 1)));
        s
    } else {
        format!("{icon} {text}")
    }
}
