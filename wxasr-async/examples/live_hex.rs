//! hex 后端的实时桌面听写 CLI(wxasr-provider `live_desktop` 的 hex 版):
//! 采集桌面回环(系统正在播放的声音),hex-asr(GGUF/Vulkan)识别。
//!
//! `cargo run --release -p wxasr-async --features hex --example live_hex -- [whisper|qwen|模型路径]`
//!
//! - 模型:`qwen`(默认,Qwen3-ASR-0.6B,小而快)/ `whisper`
//!   (whisper-large-v3-turbo)/ 任意 `.gguf` 路径。模型目录默认
//!   `G:\env\model\asr`,可用 `HEX_MODELS_DIR` 覆盖
//! - 设备默认 `Auto`(Vulkan GPU 优先),`HEXASR_DEVICE=cpu` 强制 CPU
//!   (whisper-large 在 CPU 上会超时,选它请用 GPU)
//! - 中间结果(💬)是流式投机文本,定稿(✅)携带完整 clip;hex 语义一次
//!   会话恰一个 Final,Final 的语言识别(language)有则显示
//! - Ctrl+C 定稿后退出;再按一次强制退出

#[cfg(feature = "hex")]
use crossterm::cursor::{Hide, MoveTo, Show};
#[cfg(feature = "hex")]
use crossterm::execute;
#[cfg(feature = "hex")]
use crossterm::terminal::{size, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen};

#[cfg(feature = "hex")]
#[tokio::main]
async fn main() {
    use std::sync::Arc;

    use wxasr_async::backend_hex::HexBackend;
    use wxasr_async::{start_with, AsrEvent, AudioIn, Config};
    use wxasr_provider::{AudioSource, DesktopAudio};

    let model = resolve_model(std::env::args().nth(1));
    if !std::path::Path::new(&model).is_file() {
        panic!("模型不存在: {model}(目录默认 G:\\env\\model\\asr,可用 HEX_MODELS_DIR 覆盖)");
    }
    let device = match std::env::var("HEXASR_DEVICE").as_deref() {
        Ok("cpu") => hex_asr::Device::Cpu,
        Ok("gpu") => hex_asr::Device::Gpu,
        _ => hex_asr::Device::Auto,
    };

    eprintln!("加载 {model} …");
    let started = std::time::Instant::now();
    let engine = Arc::new(
        hex_asr::Engine::load(&hex_asr::ModelSpec::new(&model), device)
            .unwrap_or_else(|error| panic!("模型加载失败: {error:#}")),
    );
    eprintln!(
        "就绪(device={device:?},加载 {:.1}s)。播放任意声音即可;Ctrl+C 退出",
        started.elapsed().as_secs_f32()
    );

    let (asr, mut rx) = start_with(
        Box::new(HexBackend::new(engine)),
        Config {
            rotate: wxasr_async::policy::RotatePolicy::Auto {
                soft_ms: 10,
                hard_ms: 20,
            },
            partial: true,
            partial_min_fresh: 30,
            event_capacity: 64,
            thread_name: None,
        },
        AudioIn::MONO_16K,
    );
    let sink = {
        let asr = asr.clone();
        let sink: wxasr_provider::PcmSink = Arc::new(move |pcm: &[i16]| asr.push(pcm));
        sink
    };
    let mut asr_fin = Some(asr);

    let source = DesktopAudio::new();
    let mut capture = source.start(None, sink).expect("桌面采集启动失败");

    let mut out = std::io::stdout();
    let _ = execute!(out, EnterAlternateScreen, Hide);

    let mut committed: Vec<String> = Vec::new(); // Final 定稿各行
    let mut live = String::new(); // 当前段中间结果
    let mut finishing = false;
    loop {
        render(&mut out, &committed, &live);
        tokio::select! {
            ev = rx.recv() => match ev {
                Some(AsrEvent::Partial(t)) => live = t.content,
                Some(AsrEvent::Final(t)) => {
                    let lang = t.language.map(|l| format!(" [{l}]")).unwrap_or_default();
                    committed.push(format!("{}{lang}", t.content));
                    live.clear();
                }
                None => break, // 通道关闭:会话结束
            },
            _ = tokio::signal::ctrl_c() => {
                if finishing {
                    break; // 第二次 Ctrl+C 强制退出
                }
                finishing = true;
                capture.stop();
                if let Some(a) = asr_fin.take() {
                    a.finish(); // 定稿当前段,等通道关闭
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

#[cfg(not(feature = "hex"))]
fn main() {
    eprintln!("live_hex 需要 hex 后端:cargo run --features hex --example live_hex");
}

#[cfg(feature = "hex")]
fn resolve_model(arg: Option<String>) -> String {
    let dir = std::env::var("HEX_MODELS_DIR").unwrap_or_else(|_| "G:\\env\\model\\asr".to_string());
    let join = |name: &str| format!("{dir}\\{name}");
    match arg.as_deref() {
        None | Some("qwen") => join("Qwen3-ASR-0.6B-Q8_0.gguf"),
        Some("whisper") => join("whisper-large-v3-turbo-Q8_0.gguf"),
        Some(path) => path.to_string(),
    }
}

/// 回到左上角、清掉以下所有行,逐行重画:✅ 定稿行在前、💬 当前行在末尾。
/// 每行按 CJK 双列保守折算截断到不折行,整体只保留最后一屏,避免滚动残影
#[cfg(feature = "hex")]
fn render(out: &mut impl std::io::Write, committed: &[String], live: &str) {
    let (cols, rows) = size().unwrap_or((80, 24));
    let width = cols as usize / 2 - 2;
    let mut lines: Vec<String> = committed.iter().map(|l| fmt_line("✅", l, width)).collect();
    if !live.is_empty() {
        lines.push(fmt_line("💬", live, width));
    }
    if lines.is_empty() {
        lines.push(String::from(
            "🎙 hex 实时桌面听写中(播放任意声音即可)… Ctrl+C 退出",
        ));
    }
    let lines = &lines[lines.len().saturating_sub(rows as usize)..];
    let _ = execute!(out, MoveTo(0, 0), Clear(ClearType::FromCursorDown));
    for l in lines {
        let _ = writeln!(out, "{l}");
    }
    let _ = out.flush();
}

/// icon + 单行文本,超宽时只显示尾部(前缀 …),保证物理上不折行
#[cfg(feature = "hex")]
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
