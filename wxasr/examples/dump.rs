//! 打印模型目录内 .xnet 容器的张量清单。
//!
//! 用法：`cargo run --release --example dump -- <模型目录>`

use memmap2::Mmap;
use wxasr::xnet;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "G:/codes/rust/project-a/model".to_string());
    let path = std::fs::read_dir(&dir)
        .expect("目录不可读")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|e| e == "xnet"))
        .expect("目录中无 .xnet 文件");
    let map = unsafe { Mmap::map(&std::fs::File::open(&path).unwrap()).unwrap() };
    let t0 = std::time::Instant::now();
    let recs = xnet::parse(&map).unwrap();
    println!(
        "解析 {} 条记录，耗时 {:.0}ms，文件 {:.1}MB\n",
        recs.len(),
        t0.elapsed().as_secs_f64() * 1e3,
        map.len() as f64 / 1e6
    );

    let mut names: Vec<&String> = recs.keys().collect();
    names.sort();
    let (mut f32c, mut i8c, mut q) = (0usize, 0usize, 0usize);
    for name in &names {
        let r = &recs[name.as_str()];
        match r.dtype {
            xnet::F32 => f32c += 1,
            xnet::I8 => {
                i8c += 1;
                if r.scale.is_some() {
                    q += 1;
                }
            }
            _ => {}
        }
        println!(
            "{name:<64} dtype={:<2} dims={:<20} scale={:<10} off={}",
            r.dtype,
            format!("{:?}", r.dims),
            r.scale.map(|s| s.to_string()).unwrap_or_default(),
            r.off
        );
    }
    println!("\n总计：{} f32, {} i8（其中 {q} 条带量化 scale）", f32c, i8c);
}
