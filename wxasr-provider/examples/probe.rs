//! 桌面回环（loopback）可用性探测。
//!
//! 在默认输出设备上播放 440Hz 音，同时尝试在其上建立输入流，
//! 统计收到的样本数与峰值，验证采集通路。
//!
//! `cargo run --release --example probe -- [秒数]`

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::time::Duration;

fn main() {
    let secs: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let host = cpal::default_host();
    println!("音频 host: {}", host.id().name());
    let dev = host.default_output_device().expect("没有输出设备");
    let name = dev.name().unwrap_or_default();
    let cfg = dev.default_output_config().expect("读取输出配置失败");
    let rate = cfg.sample_rate().0;
    let ch = cfg.channels();
    println!(
        "输出设备: {name}  {rate}Hz {ch}ch {:?}",
        cfg.sample_format()
    );

    // 1) 播放 440Hz 音（振幅 0.3）
    let mut phase = 0.0f32;
    let tone = dev
        .build_output_stream(
            &cfg.clone().into(),
            move |d: &mut [f32], _| {
                for frame in d.chunks_mut(ch as usize) {
                    phase += 2.0 * std::f32::consts::PI * 440.0 / rate as f32;
                    let v = phase.sin() * 0.3;
                    for s in frame {
                        *s = v;
                    }
                }
            },
            |_| {},
            None,
        )
        .expect("建立播放流失败");
    tone.play().unwrap();
    println!("正在播放 440Hz…");

    // 2) 在输出设备上建输入流（WASAPI loopback），统计样本数与峰值
    let total = std::sync::Arc::new(AtomicU64::new(0));
    let peak = std::sync::Arc::new(AtomicI32::new(0));
    let (t2, p2) = (total.clone(), peak.clone());
    let loopback = dev.build_input_stream(
        &cfg.clone().into(),
        move |d: &[f32], _| {
            t2.fetch_add(d.len() as u64, Ordering::Relaxed);
            let m = d
                .iter()
                .map(|v| (v.abs() * 32767.0) as i32)
                .max()
                .unwrap_or(0);
            p2.fetch_max(m, Ordering::Relaxed);
        },
        |e| eprintln!("输入流错误: {e}"),
        None,
    );

    match loopback {
        Ok(stream) => {
            stream.play().unwrap();
            std::thread::sleep(Duration::from_secs(secs));
            drop(stream);
            drop(tone);
            let n = total.load(Ordering::Relaxed);
            let p = peak.load(Ordering::Relaxed);
            println!("\nloopback 通路: OK");
            println!(
                "  {secs}s 收到 {n} 个样本（{:.0}/s，期望约 {}）",
                n / secs,
                rate * ch as u32
            );
            println!("  峰值 {p}/32767（>2000 说明真的捕获到了播放声）");
        }
        Err(e) => {
            println!("\nloopback 通路: 失败 — {e}");
        }
    }
}
