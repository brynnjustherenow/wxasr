# 生成集成测试用的中文 TTS wav（16kHz/单声道/16bit）+ 同名文本。
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gen_tts.ps1
param(
    [string]$OutDir = "$PSScriptRoot\..\tests\data"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$zh = $synth.GetInstalledVoices() | Where-Object {
    $_.VoiceInfo.Culture.Name -like "zh-*"
}
if (-not $zh) {
    Write-Error "未找到中文语音（需安装 zh-CN TTS，如 Microsoft Huihui）"
}
$synth.SelectVoice($zh[0].VoiceInfo.Name)
Write-Host "使用语音: $($zh[0].VoiceInfo.Name)"

$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

$clips = @(
    @{ f = "tts_short_1.wav"; t = "今天天气真不错，我们一起去公园散步吧。" },
    @{ f = "tts_short_2.wav"; t = "请帮我打开客厅的灯，再把空调调到二十六度。" },
    @{ f = "tts_short_3.wav"; t = "明天早上八点有一个重要会议，记得提醒我带上笔记本电脑和充电器。" },
    @{ f = "tts_long_1.wav";  t = "语音识别技术在过去几年里取得了很大的进步。从最早只能识别孤立的词语，到现在可以连续识别自然语言，背后是深度学习模型的不断发展。离线识别让用户不必把语音上传到服务器，既保护了隐私，也节省了流量。这个测试句子的长度超过二十秒，用来验证流式识别在多次定稿之后的最终效果是否稳定可靠，请耐心听完这一整段话。" }
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
foreach ($c in $clips) {
    $wav = Join-Path $OutDir $c.f
    $txt = [IO.Path]::ChangeExtension($wav, ".txt")
    $synth.SetOutputToWaveFile($wav, $fmt)
    $synth.Speak($c.t)
    $synth.SetOutputToNull()
    [IO.File]::WriteAllText($txt, $c.t, [Text.UTF8Encoding]::new($false))
    $len = (Get-Item $wav).Length
    Write-Host ("{0}  {1:N0} 字节  ({2:N1}s)" -f $c.f, $len, ($len / 32000.0))
}
Write-Host "完成，输出目录: $OutDir"
