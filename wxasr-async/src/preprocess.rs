//! 调用方音频 → 后端原生格式的流式预处理。
//!
//! 统一层不写死采样率:调用方在 [`crate::AudioIn`] 里声明自己的格式,
//! [`InputConverter`] 在会话内部把交错的任意采样率/声道 i16 逐步转换
//! 成 16 kHz 单声道 i16(wxasr 与 gguf 两条后端链的公共输入),调用方
//! 无感。16 kHz 单声道本身就是目标格式,直通零开销。

use crate::AudioIn;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler as _, WindowFunction};

/// 16 kHz:两条后端链的公共采样率。
const TARGET_RATE: u32 = 16_000;
/// 重采样器的输入块:一次 1024 个源样本。
const CHUNK_SIZE: usize = 1024;

/// 把一段声明的 i16 流式转换成 16 kHz 单声道 i16。
///
/// `push` 可能返回空(重采样还在攒输入块);会话结束前调用
/// [`InputConverter::flush`] 取出残留。16 kHz 单声道直通。
pub struct InputConverter {
    channels: usize,
    /// `None`:采样率已是 16 kHz,只降混不重采样
    resampler: Option<Resampler>,
}

impl InputConverter {
    /// 按声明的格式创建。失败仅当采样率无法重采样到 16 kHz。
    pub fn new(audio: AudioIn) -> Result<Self, String> {
        let channels = usize::from(audio.channels.max(1));
        let resampler = (audio.sample_rate != TARGET_RATE)
            .then(|| Resampler::try_new(audio.sample_rate))
            .transpose()?;
        Ok(Self {
            channels,
            resampler,
        })
    }

    /// 是否直通(16 kHz 单声道):调用方可以跳过转换本身。
    pub fn bypassed(&self) -> bool {
        self.channels == 1 && self.resampler.is_none()
    }

    /// 转换一段;输出为 16 kHz 单声道 i16(可能为空,直到攒够输入)。
    pub fn push(&mut self, pcm: &[i16]) -> Vec<i16> {
        if self.bypassed() {
            return pcm.to_vec();
        }
        let mono = downmix_f32(pcm, self.channels);
        let converted = match &mut self.resampler {
            Some(resampler) => resampler.push(&mono),
            None => mono,
        };
        to_i16(&converted)
    }

    /// 取出重采样残留(会话结束时调用一次;直通时为空)。
    pub fn flush(&mut self) -> Vec<i16> {
        match self.resampler.take() {
            Some(resampler) => to_i16(&resampler.finish()),
            None => Vec::new(),
        }
    }
}

/// 交错 i16 → 单声道 f32:多声道取平均,单声道即转格式。
fn downmix_f32(pcm: &[i16], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return pcm
            .iter()
            .map(|&sample| f32::from(sample) / 32_768.0)
            .collect();
    }
    let frames = pcm.len() / channels;
    (0..frames)
        .map(|frame| {
            let frame = &pcm[frame * channels..][..channels];
            frame.iter().map(|&sample| f32::from(sample)).sum::<f32>()
                / (channels as f32 * 32_768.0)
        })
        .collect()
}

/// f32 → i16,削波在先。
fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&sample| (sample.clamp(-1.0, 1.0) * 32_767.0).round() as i16)
        .collect()
}

/// 重采样一个单声道块;partial 块用静音补齐,所以空 partial 块冲出
/// 重采样器的延迟。
fn resample_chunk(resampler: &mut Fft<f32>, samples: &[f32], partial: bool) -> Vec<f32> {
    let input =
        InterleavedSlice::new(samples, 1, samples.len()).expect("mono chunk fits its slice");
    let indexing = partial.then(|| Indexing {
        partial_len: Some(samples.len()),
        ..Indexing::default()
    });
    resampler
        .process(&input, indexing.as_ref())
        .expect("fixed-rate resampling must succeed")
        .take_data()
}

/// 任意采样率 → 16 kHz 的流式重采样(gguf-asr `audio::Resampler` 的
/// 同源实现):输出先扣除重采样延迟,`finish` 后的总长与整段转换一致。
struct Resampler {
    source_rate: u32,
    resampler: Fft<f32>,
    input: Vec<f32>,
    source_samples: usize,
    delay_remaining: usize,
    produced: usize,
}

impl Resampler {
    /// 失败仅当 `source_rate` 无法转换到 16 kHz。
    fn try_new(source_rate: u32) -> Result<Self, String> {
        let resampler = Fft::new_custom(
            source_rate as usize,
            TARGET_RATE as usize,
            CHUNK_SIZE,
            1,
            1,
            WindowFunction::BlackmanHarris2,
            FixedSync::Input,
        )
        .map_err(|error| error.to_string())?;
        let delay_remaining = resampler.output_delay();
        Ok(Self {
            source_rate,
            resampler,
            input: Vec::with_capacity(CHUNK_SIZE),
            source_samples: 0,
            delay_remaining,
            produced: 0,
        })
    }

    /// 喂入连续的源样本;返回它们产出的 16 kHz 输出(攒够输入块前
    /// 可能为空)。
    fn push(&mut self, mut samples: &[f32]) -> Vec<f32> {
        self.source_samples += samples.len();
        let mut output = Vec::new();
        while !samples.is_empty() {
            let take = samples.len().min(CHUNK_SIZE - self.input.len());
            self.input.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.input.len() == CHUNK_SIZE {
                output.extend(resample_chunk(&mut self.resampler, &self.input, false));
                self.input.clear();
            }
        }
        self.trim_delay(output)
    }

    /// 冲出残留;总长裁到精确的换算长度。
    fn finish(mut self) -> Vec<f32> {
        let expected = self.source_samples * TARGET_RATE as usize / self.source_rate as usize;
        let mut output = Vec::new();
        if !self.input.is_empty() {
            let flushed = resample_chunk(&mut self.resampler, &self.input, true);
            output.extend(self.trim_delay(flushed));
        }
        for _ in 0..8 {
            if self.produced >= expected {
                break;
            }
            let flushed = resample_chunk(&mut self.resampler, &[], true);
            output.extend(self.trim_delay(flushed));
        }
        let excess = self.produced.saturating_sub(expected);
        output.truncate(output.len().saturating_sub(excess));
        output
    }

    fn trim_delay(&mut self, mut output: Vec<f32>) -> Vec<f32> {
        let trimmed = self.delay_remaining.min(output.len());
        output.drain(..trimmed);
        self.delay_remaining -= trimmed;
        self.produced += output.len();
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(samples: usize, frequency_hz: f32, rate: u32) -> Vec<i16> {
        (0..samples)
            .map(|i| {
                (0.4 * (i as f32 * frequency_hz * std::f32::consts::TAU / rate as f32).sin()
                    * 32_767.0) as i16
            })
            .collect()
    }

    fn convert_all(audio: AudioIn, pcm: &[i16]) -> Vec<i16> {
        let mut converter = InputConverter::new(audio).unwrap();
        let mut out = converter.push(pcm);
        out.extend(converter.flush());
        out
    }

    #[test]
    fn mono_16k_passes_through_unchanged() {
        let pcm = tone(4_000, 440.0, 16_000);
        let mut converter = InputConverter::new(AudioIn::MONO_16K).unwrap();
        assert!(converter.bypassed());
        assert_eq!(converter.push(&pcm), pcm);
        assert!(converter.flush().is_empty());
    }

    #[test]
    fn downsampling_48k_reaches_the_exact_length() {
        let pcm = tone(48_000 * 2, 440.0, 48_000);
        let converted = convert_all(
            AudioIn {
                sample_rate: 48_000,
                channels: 1,
            },
            &pcm,
        );
        assert_eq!(converted.len(), pcm.len() / 3, "48k → 16k 是 3:1");
    }

    #[test]
    fn upsampling_8k_reaches_the_exact_length() {
        let pcm = tone(8_000 * 2, 220.0, 8_000);
        let converted = convert_all(
            AudioIn {
                sample_rate: 8_000,
                channels: 1,
            },
            &pcm,
        );
        assert_eq!(converted.len(), pcm.len() * 2, "8k → 16k 是 1:2");
    }

    /// 44.1 kHz:真实麦克风最常见的采样率,非整数比。
    #[test]
    fn resampling_44_1k_the_common_mic_rate_works() {
        let pcm = tone(44_100, 440.0, 44_100);
        let converted = convert_all(
            AudioIn {
                sample_rate: 44_100,
                channels: 1,
            },
            &pcm,
        );
        assert_eq!(converted.len(), 16_000, "1 s of 44.1k is 1 s of 16k");
    }

    /// 增量转换与整段转换一致:任意 push 分块不影响结果。
    #[test]
    fn incremental_chunks_match_the_whole_conversion() {
        let audio = AudioIn {
            sample_rate: 48_000,
            channels: 1,
        };
        let pcm = tone(48_000 * 2, 440.0, 48_000);
        let whole = convert_all(audio, &pcm);
        let mut converter = InputConverter::new(audio).unwrap();
        let mut incremental = Vec::new();
        for chunk in pcm.chunks(977) {
            incremental.extend(converter.push(chunk));
        }
        incremental.extend(converter.flush());
        assert_eq!(incremental.len(), whole.len());
        assert!(
            incremental
                .iter()
                .zip(whole.iter())
                .all(|(a, b)| (a - b).abs() <= 1),
            "增量与整段转换一致(±1 量化误差)"
        );
    }

    /// 立体声降混:左右声道取平均。
    #[test]
    fn stereo_downmixes_to_the_channel_average() {
        let left: Vec<i16> = (0..1_000).map(|i| (i % 100) as i16).collect();
        let right: Vec<i16> = (0..1_000).map(|i| (i % 100) as i16 * -2).collect();
        let interleaved: Vec<i16> = left
            .iter()
            .zip(right.iter())
            .flat_map(|(&l, &r)| [l, r])
            .collect();
        let converted = convert_all(
            AudioIn {
                sample_rate: 16_000,
                channels: 2,
            },
            &interleaved,
        );
        assert_eq!(converted.len(), 1_000);
        for (frame, (&l, &r)) in converted.iter().zip(left.iter().zip(right.iter())) {
            let expected = ((l as i32 + r as i32) / 2).clamp(-32_768, 32_767) as i16;
            assert_eq!(*frame, expected, "frame averages its channels");
        }
    }
}
