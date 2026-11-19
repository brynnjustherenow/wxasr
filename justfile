# project-a 常用命令(`just --list` 看全部)。Windows 下走 PowerShell;
# 凡带 gguf 的配方需要 CMake + Vulkan SDK(见 AGENTS.md「构建环境」)。

set windows-shell := ["powershell", "-NoLogo", "-Command"]
# gguf 构建环境四件套:装完工具链的已存进程拿不到机器环境变量,必须显式设
gguf_env := '$env:Path = "C:\Program Files\CMake\bin;$env:Path"; $env:VULKAN_SDK = "C:\VulkanSDK\1.4.363.0"; $env:LIB = "C:\VulkanSDK\1.4.363.0\Lib;$env:LIB"; $env:INCLUDE = "C:\VulkanSDK\1.4.363.0\Include;$env:INCLUDE";'

# 列出全部配方
default:
    @just --list

# ── 构建 ──────────────────────────────────────────────

# 构建(无 gguf feature,不需要 CMake/Vulkan)
build:
    cargo build --workspace --examples

# 构建 gguf 后端(wxasr-async --features gguf + 示例)
build-gguf:
    {{gguf_env}} cargo build -p wxasr-async --features gguf --examples

# ── 检查与测试 ────────────────────────────────────────

# fmt + clippy + 测试(无 gguf feature)
check:
    cargo fmt --all
    cargo clippy --workspace --all-targets
    cargo test -p wxasr-async
    cargo test -p wxasr-provider

# 同 check,再含 gguf feature 与 gguf-asr(需要 CMake + Vulkan SDK)
check-gguf: check
    {{gguf_env}} cargo clippy -p wxasr-async --all-targets --features gguf
    {{gguf_env}} cargo test -p wxasr-async --features gguf
    {{gguf_env}} cargo test -p gguf-asr

# 真实模型门控测试:just test-model parakeet(流式轮换 + 伪流式反向升级)/ just test-model qwen(伪流式中文)
# / just test-model whisper(伪流式英语 wav 验收)/ just test-model <gguf路径>
# --test-threads=1:多个 36s 实时会话并行会争抢同一 GPU 引擎,输家流 decode 落后 → 0 partial(假失败)
test-model model="qwen":
    {{gguf_env}} $m = if ("{{model}}" -eq "whisper") { "$env:LOCALAPPDATA\hex\models\whisper-large-v3-turbo-Q8_0.gguf" } elseif ("{{model}}" -eq "qwen") { "$env:LOCALAPPDATA\hex\models\Qwen3-ASR-0.6B-Q8_0.gguf" } elseif ("{{model}}" -eq "parakeet") { "G:\env\model\asr\parakeet-unified-en-0.6b-Q8_0.gguf" } else { "{{model}}" }; $env:GGUFASR_MODEL = $m; $env:GGUFASR_WAV = "$PWD\wxasr-async\tests\data\sapi_en.wav"; "GGUFASR_MODEL = $m"; cargo test -p gguf-asr --test model -- --nocapture --test-threads=1; cargo test -p wxasr-async --features gguf --test gguf_backend -- --nocapture --test-threads=1

# ── 实时听写示例 ──────────────────────────────────────

# wxasr 后端实时桌面听写(播放任意声音即可看到转写)
live model_dir="./model":
    cargo run --release -p wxasr-provider --example live_desktop -- {{model_dir}}

# gguf 后端实时桌面听写(伪流式开启:任何模型都有💬/✅/🏁):just live-gguf parakeet(真流式)
# / just live-gguf whisper / just live-gguf qwen(伪流式:💬 整段重转写可回改)/ just live-gguf <gguf路径>
# 设备默认 Auto(Vulkan GPU 优先);GGUFASR_DEVICE=cpu just live-gguf qwen 强制 CPU(whisper 用 CPU 会撞引擎 deadline,属设计内)
live-gguf model="qwen":
    {{gguf_env}} cargo run --release -p wxasr-async --features gguf --example live_gguf -- {{model}}

# 产出两个独立听写应用(编译期烘焙默认模型,无参数直接运行):
# target/release/live_gguf_whisper.exe(含防幻觉 detector)/ live_gguf_qwen.exe
dist-gguf:
    {{gguf_env}} $env:GGUF_LIVE_MODEL = "whisper"; cargo build --release -p wxasr-async --features gguf --example live_gguf; Copy-Item target\release\examples\live_gguf.exe target\release\live_gguf_whisper.exe -Force
    {{gguf_env}} $env:GGUF_LIVE_MODEL = "qwen"; cargo build --release -p wxasr-async --features gguf --example live_gguf; Copy-Item target\release\examples\live_gguf.exe target\release\live_gguf_qwen.exe -Force
