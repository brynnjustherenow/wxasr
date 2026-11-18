//! hex 后端的实时桌面听写 CLI(wxasr-provider `live_desktop` 的 hex 版):
//! 采集桌面回环(系统正在播放的声音),hex-asr(GGUF/Vulkan)识别。
//!
//! `cargo run --release -p wxasr-async --features hex --example live_hex -- [parakeet|whisper|qwen|模型路径]`
//!
//! 无参数运行时用编译期烘焙的默认模型(`HEX_LIVE_MODEL`),据此产出两个
//! 独立应用:`HEX_LIVE_MODEL=whisper` → `live_hex_whisper.exe`,
//! `HEX_LIVE_MODEL=qwen` → `live_hex_qwen.exe`(见 justfile `dist-hex`)。
//!
//! - 模型:`parakeet`(Parakeet Unified English,真流式)/ `qwen`
//!   (Qwen3-ASR-0.6B)/ `whisper`(whisper-large-v3-turbo)/ 任意
//!   `.gguf` 路径。模型目录默认 `G:\env\model\asr`,可用 `HEX_MODELS_DIR`
//!   覆盖;不自报语言检测的模型用 `HEXASR_LANG=<ISO>` 给提示(parakeet
//!   别名已自带 en)
//! - 设备默认 `Auto`(Vulkan GPU 优先),`HEXASR_DEVICE=cpu` 强制 CPU
//!   (whisper-large 在 CPU 上会超时,选它请用 GPU)
//! - 伪流式始终开启(`HexBackend::with_pseudo`):**任何模型都有完整
//!   事件行为**——流式家族(parakeet)自动走真流式路径;离线模型
//!   (whisper/qwen)由伪流式补全:💬 = 每 2s 整段重转写(**可回改**,
//!   阻塞期间音频排队不丢),✅/🏁 = 段权威转写(后向延迟 = 段长/
//!   推理速度,伪流式的固有代价)
//! - 事件:💬 Partial(投机文本,可回改)/ ✅ Commit(轮换定稿,
//!   会话继续)/ 🏁 End(结束的最后一段,随后通道关闭)。定稿行尾带
//!   元数据后缀 `[langid · spkN · 起–止s · N段]`(后端给什么显什么,
//!   时间戳为会话绝对时刻)。轮换本应用
//!   固定 Custom 版 Auto 逻辑:soft 10s / 静音门 300ms / hard 30s 兜底
//!   ——切点尽量落在自然停顿上(policy.rs 的 Auto 门是冻结的 500ms,
//!   混语口播停顿少、硬切频落句中,故 Custom 放宽),hard 30s 恰为
//!   whisper 单窗上限——hex 后端的 Punct 切点取判定时刻(滞后 ≤~1s),
//!   会把下一句开头带进上一段,故不用;模型
//!   目录有 `silero-vad-v6.2-F32.gguf` 就自动启用 Silero VAD 做
//!   静音信号,缺文件告警退化能量门;同一文件还会接进程级 speech
//!   detector(whisper 专属防幻觉:裁掉语音段落的静音边沿、跳过
//!   长片段里的全静音 chunk,与 endpointing VAD 相互独立)
//! - Ctrl+C 定稿后退出;再按一次强制退出

#[cfg(feature = "hex")]
use crossterm::cursor::{Hide, MoveTo, Show};
#[cfg(feature = "hex")]
use crossterm::execute;
#[cfg(feature = "hex")]
use crossterm::terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, size};

#[cfg(feature = "hex")]
#[tokio::main]
async fn main() {
    use std::sync::Arc;

    use wxasr_async::backend_hex::{HexBackend, PseudoOptions};
    use wxasr_async::policy::{RotateCtx, RotatePolicy};
    use wxasr_async::{AsrEvent, AudioIn, Config, start_with};
    use wxasr_provider::{AudioSource, DesktopAudio};

    // 运行时参数优先;无参时用编译期烘焙的默认(HEX_LIVE_MODEL,产出
    // 独立应用),两者都没有则 resolve_model 落到 qwen
    let arg = std::env::args()
        .nth(1)
        .or_else(|| option_env!("HEX_LIVE_MODEL").map(str::to_string));
    let model = resolve_model(arg.as_deref());
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
        hex_asr::Engine::load(&resolve_spec(arg.as_deref(), &model), device)
            .unwrap_or_else(|error| panic!("模型加载失败: {error:#}")),
    );
    let streams = engine.streams(); // engine 进 backend 前取好
    // 防幻觉 speech detector(whisper 专属):裁掉语音段落的静音边沿、
    // 跳过长片段里的全静音 chunk——静音编字是 whisper 家族的病理,
    // qwen/parakeet 对静音本就产空,裁边只有切掉轻起音的风险没有收益,
    // 故读 GGUF 头(不加载权重)判明架构后才接。与下面的 endpointing
    // VAD(跨模型,轮换静音信号)是两个独立实例;引擎之后接:进程后端
    // 已注册,detector 的 CPU 加载不会再钉后端。
    if hex_asr::gguf_meta(&model).is_ok_and(|meta| meta.architecture == "whisper")
        && let Some(path) = vad_path()
    {
        hex_asr::speech_detector::set_model(&path);
        hex_asr::speech_detector::prewarm(); // 后台加载,首次转写不付等待
        eprintln!("防幻觉 speech detector 已接(whisper 静音边沿裁剪)");
    }
    // Silero VAD 在引擎之后加载(先引擎:进程后端已注册,VAD 不会把它钉在 CPU)
    // 伪流式始终开启:离线模型由此获得完整事件行为,流式模型自动升级
    let backend = match load_vad() {
        Some(Ok(vad)) => {
            eprintln!("Silero VAD 就绪(静音 endpointing 信号)");
            HexBackend::with_pseudo(
                engine,
                PseudoOptions {
                    vad: Some(vad),
                    ..PseudoOptions::default()
                },
            )
        }
        None => {
            eprintln!("注意:模型目录无 silero-vad-v6.2-F32.gguf,静音信号退化能量门(轮换仍可用)");
            HexBackend::pseudo_streaming(engine)
        }
        Some(Err(error)) => {
            eprintln!("注意:Silero VAD 加载失败({error:#}),静音信号退化能量门(轮换仍可用)");
            HexBackend::pseudo_streaming(engine)
        }
    };
    // 轮换:Auto 同款逻辑的 Custom 实现——soft 10s / hard 30s / 静音门
    // 300ms。不用 policy.rs 的 Auto:其 500ms 门是冻结常量(红线,policy.rs
    // 零改动),而实听混语口播里 500ms 停顿太少 → 软门够不着,20s 硬切
    // 频繁落在句中;门调到 300ms 让切点尽量落在自然停顿上,硬切放宽到
    // 30s(= whisper 单窗上限,伪流式段整段转写仍单窗)减少硬切次数。
    // 硬切本身仍兜底(连续说话无停顿),只是从"每 20s 必切"退到"每 30s 必切"。
    // 不用默认的 Punct:hex 后端只有文本没有位置,标点切点=判定时刻
    // (滞后 ≤~1s),会把下一句开头带进上一段;静音切点无滞后、下一段
    // 干净从语音起(wxasr 后端的 Punct 按位置切,无此缺陷,全局默认不动)
    let rotate = RotatePolicy::Custom(Arc::new(|ctx: &RotateCtx| {
        ctx.stream_ms >= 30_000 || (ctx.stream_ms >= 10_000 && ctx.silence_ms >= 300)
    }));
    eprintln!(
        "就绪(device={device:?},流式={streams},轮换=自定义Auto{{soft 10s, hard 30s, 静音门 300ms}},加载 {:.1}s)。播放任意声音即可;Ctrl+C 退出",
        started.elapsed().as_secs_f32()
    );
    if !streams {
        eprintln!("注意:该模型在此设备不支持流式识别——走伪流式:💬 为每 2s 整段重转写(可回改),");
        eprintln!(
            "      ✅/🏁 为整段权威定稿(后向延迟 = 段长/推理速度;GPU 建议,CPU+whisper 会超引擎超时)"
        );
    }

    let (asr, mut rx) = start_with(
        Box::new(backend),
        Config {
            rotate,
            ..Config::default()
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

    let mut committed: Vec<(&'static str, String)> = Vec::new(); // 定稿各行:(✅/🏁, 文本)
    let mut live = String::new(); // 当前段中间结果
    let mut finishing = false;
    loop {
        render(&mut out, &committed, &live);
        tokio::select! {
            ev = rx.recv() => match ev {
                Some(AsrEvent::Partial(t)) => live = t.content,
                Some(AsrEvent::Commit(t)) => {
                    committed.push(("✅", with_meta(t)));
                    live.clear();
                }
                Some(AsrEvent::End(t)) => {
                    committed.push(("🏁", with_meta(t)));
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
                    a.finish(); // 定稿当前段(🏁 End),等通道关闭
                }
            }
        }
    }
    let _ = execute!(out, LeaveAlternateScreen, Show);
    capture.stop();
    for (icon, l) in &committed {
        println!("{icon} {l}");
    }
    if !live.is_empty() {
        println!("{live}"); // 强制退出时未定稿的部分
    }
}

/// 文本 + 行尾元数据后缀:语言(langid)、说话人、时间戳、分段数——
/// 后端给什么显示什么,缺失自动省略。时间戳为**会话绝对时刻**(自会话
/// 起点累计,轮换段已平移);超宽时整行自动换行完整显示。
#[cfg(feature = "hex")]
fn with_meta(t: wxasr_async::AsrText) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(lang) = &t.language {
        parts.push(lang.clone());
    }
    if let Some(speaker) = t.speaker {
        parts.push(format!("spk{speaker}"));
    }
    if !t.segments.is_empty() {
        let start = t.segments.iter().map(|s| s.start_ms).min().unwrap_or(0);
        let end = t.segments.iter().map(|s| s.end_ms).max().unwrap_or(0);
        parts.push(format!(
            "{}–{}s",
            start as f64 / 1000.0,
            end as f64 / 1000.0
        ));
        parts.push(format!("{}段", t.segments.len()));
    } else if !t.words.is_empty() {
        // 无分段但有词级时间戳:取词的包络
        let start = t.words.iter().map(|w| w.start_ms).min().unwrap_or(0);
        let end = t.words.iter().map(|w| w.end_ms).max().unwrap_or(0);
        parts.push(format!(
            "{}–{}s",
            start as f64 / 1000.0,
            end as f64 / 1000.0
        ));
        parts.push(format!("{}词", t.words.len()));
    }
    if parts.is_empty() {
        t.content
    } else {
        format!("{} [{}]", t.content, parts.join(" · "))
    }
}

#[cfg(all(test, feature = "hex"))]
mod tests {
    use super::with_meta;
    use wxasr_async::{AsrSegment, AsrText};

    #[test]
    fn meta_suffix_shows_what_the_backend_gives() {
        // 什么都没有:原文本
        assert_eq!(with_meta(AsrText::plain("你好")), "你好");
        // 全家桶:langid · 说话人 · 时间戳包络 · 段数
        let full = AsrText {
            content: "你好。".into(),
            language: Some("zh".into()),
            speaker: Some(1),
            segments: vec![
                AsrSegment {
                    start_ms: 0,
                    end_ms: 1200,
                    text: "你".into(),
                    speaker: Some(1),
                },
                AsrSegment {
                    start_ms: 1200,
                    end_ms: 3400,
                    text: "好。".into(),
                    speaker: Some(1),
                },
            ],
            words: Vec::new(),
        };
        assert_eq!(with_meta(full), "你好。 [zh · spk1 · 0–3.4s · 2段]");
    }

    #[test]
    fn char_width_cjk_is_two() {
        use super::char_width;
        assert_eq!(char_width('你'), 2, "你 U+4F60 应为双列");
        assert_eq!(char_width('a'), 1);
    }

    #[test]
    fn wrap_line_keeps_every_character() {
        use super::wrap_line;
        // 短行不折
        assert_eq!(wrap_line("✅", "你好", 80), vec!["✅ 你好"]);
        // CJK 双列:宽 8 = 前缀 4 + 2 字,续行缩进 4 列,一个字不少
        assert_eq!(
            wrap_line("✅", "你好世界再见", 8),
            vec!["✅ 你好", "    世界", "    再见"]
        );
        // ASCII 单列:宽 8 = 前缀 4 + 4 字符
        assert_eq!(wrap_line("✅", "abcdefgh", 8), vec!["✅ abcd", "    efgh"]);
        // 混排:宽度按列数而非字符数
        assert_eq!(wrap_line("✅", "aa你你", 8), vec!["✅ aa你", "    你"]);
    }
}

/// 模型目录里的 Silero VAD 文件路径(`silero-vad-v6.2-F32.gguf`,
/// 与模型目录约定一致),缺文件为 `None`。endpointing VAD 与 speech
/// detector 共用这一个文件。
#[cfg(feature = "hex")]
fn vad_path() -> Option<String> {
    let path = format!("{}\\silero-vad-v6.2-F32.gguf", models_dir());
    std::path::Path::new(&path).is_file().then_some(path)
}

/// 加载 endpointing 用的 Silero VAD。`None` = 缺文件;`Some(Err)` =
/// 文件在但加载失败——两者都告警退化能量门
#[cfg(feature = "hex")]
fn load_vad() -> Option<eyre::Result<hex_asr::SileroVad>> {
    vad_path().map(|path| hex_asr::SileroVad::load(&path))
}

#[cfg(not(feature = "hex"))]
fn main() {
    eprintln!("live_hex 需要 hex 后端:cargo run --features hex --example live_hex");
}

#[cfg(feature = "hex")]
fn models_dir() -> String {
    std::env::var("HEX_MODELS_DIR").unwrap_or_else(|_| "G:\\env\\model\\asr".to_string())
}

#[cfg(feature = "hex")]
fn resolve_model(arg: Option<&str>) -> String {
    let join = |name: &str| format!("{}\\{name}", models_dir());
    match arg {
        None | Some("qwen") => join("Qwen3-ASR-0.6B-Q8_0.gguf"),
        Some("whisper") => join("whisper-large-v3-turbo-Q8_0.gguf"),
        Some("parakeet") => join("parakeet-unified-en-0.6b-Q8_0.gguf"),
        Some(path) => path.to_string(),
    }
}

/// 模型的加载 spec:`parakeet` 别名带 en 提示(该模型不自报语言检测,
/// `ModelSpec::new` 的自动检测会被拒载),并启用它需要的 400ms trailing
/// silence。`HEXASR_LANG=<ISO>` 显式覆盖任意模型的语言提示。
#[cfg(feature = "hex")]
fn resolve_spec(arg: Option<&str>, path: &str) -> hex_asr::ModelSpec {
    match std::env::var("HEXASR_LANG")
        .ok()
        .filter(|lang| !lang.is_empty())
    {
        Some(lang) => hex_asr::ModelSpec {
            language: Some(lang),
            ..hex_asr::ModelSpec::new(path)
        },
        None => match arg {
            Some("parakeet") => hex_asr::ModelSpec::unified_english(path),
            _ => hex_asr::ModelSpec::new(path),
        },
    }
}

/// 回到左上角、清掉以下所有行,逐行重画:定稿行(✅/🏁)在前、💬 当前行
/// 在末尾。每条定稿按显示宽度**自动换行完整显示**(CJK 双列/ASCII 单列
/// 折算,续行缩进对齐正文;不再截断丢字),整体只保留最后一屏,避免
/// 滚动残影;退出备用屏后的逐行打印天然由控制台折行,同样完整
#[cfg(feature = "hex")]
fn render(out: &mut impl std::io::Write, committed: &[(&'static str, String)], live: &str) {
    let (cols, rows) = size().unwrap_or((80, 24));
    let width = cols as usize - 1; // 留 1 列余量,防个别终端最后一列折行
    let mut lines: Vec<String> = committed
        .iter()
        .flat_map(|(icon, text)| wrap_line(icon, text, width))
        .collect();
    if !live.is_empty() {
        lines.extend(wrap_line("💬", live, width));
    }
    if lines.is_empty() {
        lines.push(String::from(
            "🎙 hex 实时桌面听写中(播放任意声音即可)… Ctrl+C 退出",
        ));
    }
    let lines = &lines[lines.len().saturating_sub(rows as usize)..];
    let _ = execute!(out, MoveTo(0, 0), Clear(ClearType::FromCursorDown));
    for l in lines {
        let _ = writeln!(out, "{}", l.trim_end());
    }
    let _ = out.flush();
}

/// icon + 文本 → 按显示宽度贪心换行的物理行序列:首行 `icon `,续行
/// 4 空格缩进对齐正文。`width` 为终端列数,按 [`char_width`] 折算
#[cfg(feature = "hex")]
fn wrap_line(icon: &str, text: &str, width: usize) -> Vec<String> {
    const PREFIX: usize = 4; // "✅ " 保守按 4 列(emoji 终端宽度不稳定)
    let mut lines = Vec::new();
    let mut current = format!("{icon} ");
    let mut used = PREFIX;
    for c in text.chars() {
        let cw = char_width(c);
        if used + cw > width {
            lines.push(current);
            current = "    ".to_string();
            used = PREFIX;
        }
        current.push(c);
        used += cw;
    }
    if !current.trim_end().is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

/// 一个字符的显示列数:CJK/全角 2 列,其余 1 列(emoji 不在此算——
/// wrap_line 的前缀按固定 4 列保守处理)
#[cfg(feature = "hex")]
fn char_width(c: char) -> usize {
    let cp = c as u32;
    let wide = (0x1100..=0x115F).contains(&cp) // 谚文字母
        || (0x2E80..=0x303E).contains(&cp) // CJK 部首、注音
        || (0x3041..=0x33FF).contains(&cp) // 假名、CJK 补充
        || (0x3400..=0x4DBF).contains(&cp) // CJK 扩展 A
        || (0x4E00..=0x9FFF).contains(&cp) // CJK 统一表意
        || (0xAC00..=0xD7A3).contains(&cp) // 谚文音节
        || (0xF900..=0xFAFF).contains(&cp) // CJK 兼容表意
        || (0xFE30..=0xFE4F).contains(&cp) // CJK 兼容形式
        || (0xFF00..=0xFF60).contains(&cp) // 全角形式
        || (0xFFE0..=0xFFE6).contains(&cp)
        || (0x20000..=0x3FFFD).contains(&cp); // CJK 扩展 B+
    if wide { 2 } else { 1 }
}
