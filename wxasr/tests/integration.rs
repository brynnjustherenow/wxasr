//! 集成测试：需要真实模型目录与测试音频。
//!
//! - 模型目录：环境变量 `WXASR_MODEL`，默认 `G:/codes/rust/project-a/model`
//! - 测试音频：`tests/data/` 下 `tts_*.wav` + 同名 `.txt`（由 scripts/gen_tts.ps1 生成），
//!   以及可选的 `real_sample.wav`（无词本的真人语音冒烟样本）
//!
//! 缺任一依赖时对应测试跳过（打印提示），不失败。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use wxasr::{xnet, Model};

fn model_dir() -> Option<PathBuf> {
    let d = std::env::var("WXASR_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/model"));
    d.join("dict.decoder.utf8.txt").is_file().then_some(d)
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn load_model() -> Option<Arc<Model>> {
    let dir = model_dir().unwrap_or_else(|| {
        eprintln!("[skip] 模型目录不存在（设置 WXASR_MODEL 指向解包后的模型目录）");
        PathBuf::new()
    });
    match Model::load(&dir) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("[skip] 模型加载失败：{e}");
            None
        }
    }
}

/// 字错率（只比较字母数字字符，忽略标点与空白）
fn cer(reference: &str, hypothesis: &str) -> f64 {
    let norm = |s: &str| -> Vec<char> { s.chars().filter(|c| c.is_alphanumeric()).collect() };
    let (r, h) = (norm(reference), norm(hypothesis));
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    // Levenshtein
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

fn tts_cases() -> Vec<(PathBuf, String)> {
    let dir = data_dir();
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut wavs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("tts_") && n.ends_with(".wav"))
            })
            .collect();
        wavs.sort();
        for wav in wavs {
            let txt = wav.with_extension("txt");
            if let Ok(text) = std::fs::read_to_string(&txt) {
                out.push((wav, text.trim().to_string()));
            }
        }
    }
    out
}

#[test]
fn parser_shapes_match_expected_architecture() {
    let Some(dir) = model_dir() else {
        eprintln!("[skip] 无模型目录");
        return;
    };
    let path = dir.join("embed_140m_kld43wh_test_78000_1_merge_4.kv.conv1d.dyQu.onnx.xnet");
    let path = if path.is_file() {
        path
    } else {
        std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|e| e == "xnet"))
            .unwrap()
    };
    let map = unsafe { memmap2::Mmap::map(&std::fs::File::open(path).unwrap()).unwrap() };
    let recs = xnet::parse(&map).unwrap();

    // 量化权重带尾部 1 的三维形状 [n, k, 1]，只比较前两维
    let shape = |name: &str| -> Vec<usize> {
        let d = &recs[name].dims;
        if name.ends_with("_s8") {
            d[..2].to_vec()
        } else {
            d.clone()
        }
    };
    assert_eq!(shape("frontend.conv2d.weight"), vec![32, 1, 5, 5]);
    assert_eq!(shape("frontend.conv2d_1.weight"), vec![64, 32, 7, 7]);
    assert_eq!(shape("frontend.conv2d_2.weight"), vec![192, 64, 3, 3]);
    assert_eq!(
        shape("frontend.linear.weight_quantized_s8"),
        vec![512, 3840]
    );
    assert!(recs["frontend.linear.weight_quantized_s8"].scale.is_some());
    assert_eq!(
        shape("layers.0.attention.q_conv.weight_quantized_s8"),
        vec![512, 512]
    );
    assert_eq!(
        shape("layers.39.feed_forward.output_conv.weight_quantized_s8"),
        vec![512, 2048]
    );
    assert_eq!(shape("layers.0.prior_norm.weight"), vec![512]);
    assert_eq!(
        shape("layers.0.attention.q_conv.weight_quantized_s8"),
        vec![512, 512]
    );
    let out_n = recs["output_conv.weight_quantized_s8"].dims[0];
    assert_eq!(out_n, 10502, "输出维度应等于词表大小");
    // 每层 12 条张量 × 40 层
    let layer_recs = recs.keys().filter(|k| k.starts_with("layers.")).count();
    assert_eq!(layer_recs, 40 * 12);
    println!(
        "容器共 {} 条记录（其中 layers.* {} 条）",
        recs.len(),
        layer_recs
    );
}

#[test]
fn transcribe_tts_clips() {
    let Some(model) = load_model() else { return };
    let cases = tts_cases();
    if cases.is_empty() {
        eprintln!("[skip] tests/data 无 tts_*.wav（先运行 scripts/gen_tts.ps1）");
        return;
    }
    let mut worst = 0.0f64;
    for (wav, text) in &cases {
        let t0 = Instant::now();
        let got = model.transcribe_wav(wav).expect("转写失败");
        let dur = t0.elapsed().as_secs_f64();
        let e = cer(text, &got);
        worst = worst.max(e);
        println!(
            "[{:.1}s 耗时 {:.0}ms CER {:.1}%] {}\n           期望: {text}\n",
            wav.file_stem().unwrap().to_string_lossy(),
            dur * 1e3,
            e * 100.0,
            got
        );
        assert!(!got.trim().is_empty(), "{} 输出为空", wav.display());
        assert!(e < 0.2, "{} CER {:.1}% 超阈值", wav.display(), e * 100.0);
    }
    println!("最差 CER {:.1}%", worst * 100.0);
}

#[test]
fn streaming_matches_whole_decode_on_short_clip() {
    let Some(model) = load_model() else { return };
    let Some((wav, text)) = tts_cases()
        .into_iter()
        // 短音频 finish 才走整段重解码路径，与整段转写应严格一致
        .min_by_key(|(w, _)| w.metadata().map(|m| m.len()).unwrap_or(u64::MAX))
    else {
        eprintln!("[skip] 无测试音频");
        return;
    };
    let mut reader = hound::WavReader::open(&wav).unwrap();
    let pcm: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();

    let mut s = model.stream();
    let mut partials = Vec::new();
    for chunk in pcm.chunks(3200) {
        // 3200 样本 = 200ms
        s.push(chunk);
        if s.fresh() >= 30 {
            let p = s.partial();
            if !p.is_empty() {
                partials.push(p);
            }
        }
    }
    let final_text = s.finish();
    let whole = model.transcribe_wav(&wav).unwrap();
    // 短音频 finish 走整段重解码，应与整段转写一致
    assert_eq!(final_text, whole, "流式定稿应等于整段解码");
    println!(
        "流式: {} 个中间结果，定稿 CER {:.1}%\n  定稿: {final_text}",
        partials.len(),
        cer(&text, &final_text) * 100.0
    );
    for (i, p) in partials.iter().enumerate().take(3) {
        println!("  partial[{i}]: {p}");
    }
    assert!(!partials.is_empty(), "应有中间结果");
}

#[test]
fn long_clip_streams_and_stays_accurate() {
    let Some(model) = load_model() else { return };
    let cases = tts_cases();
    // 找最长的一条（触发分块定稿路径）
    let Some((wav, text)) = cases
        .iter()
        .max_by_key(|(w, _)| w.metadata().unwrap().len())
        .cloned()
    else {
        eprintln!("[skip] 无测试音频");
        return;
    };
    let mut reader = hound::WavReader::open(&wav).unwrap();
    let pcm: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let secs = pcm.len() as f64 / 16000.0;
    if secs < 21.0 {
        eprintln!("[skip] 最长音频 {secs:.1}s，不足以触发分块定稿（需 >20s）");
        return;
    }
    let mut s = model.stream();
    let mut partials = 0;
    let t0 = Instant::now();
    for chunk in pcm.chunks(3200) {
        s.push(chunk);
        if s.fresh() >= 30 {
            let p = s.partial();
            if !p.is_empty() {
                partials += 1;
            }
        }
    }
    let final_text = s.finish();
    let total = t0.elapsed().as_secs_f64();
    let e = cer(&text, &final_text);
    println!(
        "{secs:.1}s 音频：流式全程 {total:.2}s（实时率 {:.2}），{} 个中间结果，CER {:.1}%\n  {final_text}",
        total / secs,
        partials,
        e * 100.0
    );
    assert!(e < 0.25, "长音频流式 CER {:.1}% 超阈值", e * 100.0);
}

/// 说话中硬切（无静音可借）：对比「全切」（每次换全新流，切点后无左文）
/// 与 `rotate`（blank 对齐切点 + 携带左文与未定稿尾部接续）。跨多个
/// 切割间隔扫描，接续流总体应不差且有明确优势。
#[test]
fn rotate_carries_left_context_across_cuts() {
    let Some(model) = load_model() else { return };
    let cases = tts_cases();
    // 优先用长音频（切割次数多，统计更稳）；缺失时拼接三条短音频
    let (pcm, text) = match cases
        .iter()
        .find(|(w, _)| w.file_stem().unwrap().to_str().unwrap().starts_with("tts_long"))
    {
        Some((wav, text)) => {
            let mut r = hound::WavReader::open(wav).unwrap();
            (
                r.samples::<i16>().map(|s| s.unwrap()).collect::<Vec<_>>(),
                text.clone(),
            )
        }
        None => {
            let short: Vec<_> = cases
                .iter()
                .filter(|(w, _)| {
                    w.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("tts_short_"))
                })
                .take(3)
                .collect();
            if short.len() < 3 {
                eprintln!("[skip] 无 tts_long_*.wav 且短音频不足 3 条");
                return;
            }
            let mut pcm = Vec::new();
            let mut text = String::new();
            for (wav, t) in &short {
                let mut r = hound::WavReader::open(wav).unwrap();
                pcm.extend(r.samples::<i16>().map(|s| s.unwrap()));
                text.push_str(t);
            }
            (pcm, text)
        }
    };
    println!(
        "音频 {:.1}s，参考: {text}",
        pcm.len() as f64 / 16000.0
    );

    let mut worst_carry = 0.0f64;
    let mut sum_full = 0.0f64;
    let mut sum_carry = 0.0f64;
    for cut_s in [2.0f64, 2.5, 3.0, 4.0, 5.0, 7.0] {
        let cut = (cut_s * 16000.0) as usize;
        let run = |rotate: bool| -> String {
            let (mut s, mut fed, mut out) = (model.stream(), 0, String::new());
            for chunk in pcm.chunks(3200) {
                s.push(chunk);
                fed += chunk.len();
                if fed >= cut {
                    let (t, next) = if rotate {
                        s.rotate()
                    } else {
                        let t = s.finish();
                        (t, model.stream())
                    };
                    out.push_str(&t);
                    s = next;
                    fed = 0;
                }
            }
            out.push_str(&s.finish());
            out
        };
        let full_cut = run(false);
        let carried = run(true);
        let (e_full, e_carry) = (cer(&text, &full_cut), cer(&text, &carried));
        sum_full += e_full;
        sum_carry += e_carry;
        worst_carry = worst_carry.max(e_carry);
        println!(
            "切距 {cut_s:.1}s：全切 CER {:>5.1}%  接续流 CER {:>5.1}%",
            e_full * 100.0,
            e_carry * 100.0
        );
    }
    println!(
        "均值：全切 {:.1}% vs 接续流 {:.1}%（接续流最差 {:.1}%）",
        sum_full / 6.0 * 100.0,
        sum_carry / 6.0 * 100.0,
        worst_carry * 100.0
    );
    assert!(
        sum_carry <= sum_full,
        "接续流平均 CER ({:.1}%) 不应比全切 ({:.1}%) 差",
        sum_carry / 6.0 * 100.0,
        sum_full / 6.0 * 100.0
    );
    assert!(worst_carry < 0.15, "接续流最差 CER {:.1}% 超阈值", worst_carry * 100.0);
}

#[test]
fn rtf_under_one() {
    let Some(model) = load_model() else { return };
    let Some((wav, _)) = tts_cases().first().cloned() else {
        eprintln!("[skip] 无测试音频");
        return;
    };
    let mut reader = hound::WavReader::open(&wav).unwrap();
    let pcm: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let secs = pcm.len() as f64 / 16000.0;

    // 预热（线程池、页缓存）
    let mut f = wxasr::Fbank::new();
    let mut frames = Vec::new();
    f.push(&pcm, &mut frames);
    let _ = model.forward(&frames);

    let t0 = Instant::now();
    let ids = model.forward(&frames);
    let cost = t0.elapsed().as_secs_f64();
    println!(
        "前向 {:.0}ms / {secs:.1}s 音频 = 实时率 {:.2}（{} 帧特征 → {} 个输出帧）",
        cost * 1e3,
        cost / secs,
        frames.len(),
        ids.len()
    );
    if cfg!(debug_assertions) {
        eprintln!("[skip] debug 构建不做 RTF 断言");
        return;
    }
    assert!(cost / secs < 1.0, "实时率 {:.2} 超过 1", cost / secs);
}
