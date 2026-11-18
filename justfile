# project-a 常用命令(`just --list` 看全部)。Windows 下走 PowerShell;
# 凡带 hex 的配方需要 CMake + Vulkan SDK(见 AGENTS.md「构建环境」)。

set windows-shell := ["powershell", "-NoLogo", "-Command"]
# hex 构建环境四件套:装完工具链的已存进程拿不到机器环境变量,必须显式设
hex_env := '$env:Path = "C:\Program Files\CMake\bin;$env:Path"; $env:VULKAN_SDK = "C:\VulkanSDK\1.4.363.0"; $env:LIB = "C:\VulkanSDK\1.4.363.0\Lib;$env:LIB"; $env:INCLUDE = "C:\VulkanSDK\1.4.363.0\Include;$env:INCLUDE";'

# 列出全部配方
default:
    @just --list

# ── 构建 ──────────────────────────────────────────────

# 构建(无 hex feature,不需要 CMake/Vulkan)
build:
    cargo build --workspace --examples

# 构建 hex 后端(wxasr-async --features hex + 示例)
build-hex:
    {{hex_env}} cargo build -p wxasr-async --features hex --examples

# ── 检查与测试 ────────────────────────────────────────

# fmt + clippy + 测试(无 hex feature)
check:
    cargo fmt --all
    cargo clippy --workspace --all-targets
    cargo test -p wxasr-async
    cargo test -p wxasr-provider

# 同 check,再含 hex feature 与 hex-asr(需要 CMake + Vulkan SDK)
check-hex: check
    {{hex_env}} cargo clippy -p wxasr-async --all-targets --features hex
    {{hex_env}} cargo test -p wxasr-async --features hex
    {{hex_env}} cargo test -p hex-asr

# 真实模型门控测试:just test-model qwen / just test-model whisper / just test-model <gguf路径>
test-model model="qwen":
    {{hex_env}} $m = if ("{{model}}" -eq "whisper") { "$env:LOCALAPPDATA\hex\models\whisper-large-v3-turbo-Q8_0.gguf" } elseif ("{{model}}" -eq "qwen") { "$env:LOCALAPPDATA\hex\models\Qwen3-ASR-0.6B-Q8_0.gguf" } else { "{{model}}" }; $env:HEXASR_MODEL = $m; "HEXASR_MODEL = $m"; cargo test -p hex-asr --test model -- --nocapture --test-threads=1; cargo test -p wxasr-async --features hex --test hex_backend -- --nocapture

# ── 实时听写示例 ──────────────────────────────────────

# wxasr 后端实时桌面听写(播放任意声音即可看到转写)
live model_dir="./model":
    cargo run --release -p wxasr-provider --example live_desktop -- {{model_dir}}

# hex 后端实时桌面听写:just live-hex qwen / just live-hex whisper / just live-hex <gguf路径>
# 设备默认 Auto(Vulkan GPU 优先);HEXASR_DEVICE=cpu just live-hex qwen 强制 CPU
live-hex model="qwen":
    {{hex_env}} cargo run --release -p wxasr-async --features hex --example live_hex -- {{model}}
