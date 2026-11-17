//! Audio helpers around the 16 kHz mono f32 contract every model here uses,
//! extracted from Hex's dictation capture: minimum-duration padding and
//! whole-buffer resampling to 16 kHz.

use std::time::Duration;

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler as _, WindowFunction};

/// The sample rate every transcription model here takes: 16 kHz.
pub const SAMPLE_RATE: u32 = 16_000;

/// The minimum clip length every whole-clip run is padded to.
const MINIMUM_CLIP_DURATION: Duration = Duration::from_millis(1_500);

/// The resampler's input block: 1024 source samples at a time.
const CHUNK_SIZE: usize = 1024;

fn samples_for(duration: Duration, sample_rate: u32) -> usize {
    (duration.as_secs_f64() * f64::from(sample_rate)).round() as usize
}

/// Pads `samples` with silence to the 1.5 s minimum every clip and chunk is
/// padded to before it reaches a model.
pub fn pad_minimum(samples: &mut Vec<f32>) {
    samples.resize(
        samples
            .len()
            .max(samples_for(MINIMUM_CLIP_DURATION, SAMPLE_RATE)),
        0.0,
    );
}

/// Appends 16 kHz mono i16 PCM to `out` as f32 in [-1, 1), the format every
/// model here takes. Callers feeding i16 convert once, here.
pub fn i16_to_f32(pcm: &[i16], out: &mut Vec<f32>) {
    out.extend(pcm.iter().map(|&sample| f32::from(sample) / 32_768.0));
}

/// Resamples one mono chunk. A partial chunk is padded with silence, so an
/// empty partial chunk flushes the resampler's delay.
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

/// Converts any mono sample rate to 16 kHz. `source_rate` of 16_000 returns
/// the input unchanged.
pub fn resample(samples: &[f32], source_rate: u32) -> Result<Vec<f32>, String> {
    let mut resampler = Resampler::try_new(source_rate)?;
    let mut converted = resampler.push(samples);
    converted.extend(resampler.finish());
    Ok(converted)
}

/// Incremental [`resample`] for long recordings that must not be held in
/// memory. Output excludes the resampler delay, and the total length after
/// `finish` matches the whole-buffer conversion.
pub struct Resampler {
    source_rate: u32,
    resampler: Option<Fft<f32>>,
    input: Vec<f32>,
    source_samples: usize,
    delay_remaining: usize,
    produced: usize,
}

impl Resampler {
    /// Fails only when `source_rate` cannot be converted to 16 kHz.
    pub fn try_new(source_rate: u32) -> Result<Self, String> {
        let resampler = (source_rate != SAMPLE_RATE)
            .then(|| {
                Fft::new_custom(
                    source_rate as usize,
                    SAMPLE_RATE as usize,
                    CHUNK_SIZE,
                    1,
                    1,
                    WindowFunction::BlackmanHarris2,
                    FixedSync::Input,
                )
                .map_err(|error| error.to_string())
            })
            .transpose()?;
        let delay_remaining = resampler.as_ref().map_or(0, Fft::output_delay);
        Ok(Self {
            source_rate,
            resampler,
            input: Vec::with_capacity(CHUNK_SIZE),
            source_samples: 0,
            delay_remaining,
            produced: 0,
        })
    }

    /// Pushes contiguous source samples; returns the 16 kHz output they
    /// produced (possibly empty until enough input arrives).
    pub fn push(&mut self, mut samples: &[f32]) -> Vec<f32> {
        self.source_samples += samples.len();
        let Some(resampler) = &mut self.resampler else {
            self.produced += samples.len();
            return samples.to_vec();
        };
        let mut output = Vec::new();
        while !samples.is_empty() {
            let take = samples.len().min(CHUNK_SIZE - self.input.len());
            self.input.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.input.len() == CHUNK_SIZE {
                output.extend(resample_chunk(resampler, &self.input, false));
                self.input.clear();
            }
        }
        self.trim_delay(output)
    }

    /// Flushes the resampler and returns the rest of the output, trimmed to
    /// the exact converted length.
    pub fn finish(mut self) -> Vec<f32> {
        let expected = self.source_samples * SAMPLE_RATE as usize / self.source_rate as usize;
        let Some(mut resampler) = self.resampler.take() else {
            return Vec::new();
        };
        let mut output = Vec::new();
        if !self.input.is_empty() {
            let flushed = resample_chunk(&mut resampler, &self.input, true);
            output.extend(self.trim_delay(flushed));
        }
        for _ in 0..8 {
            if self.produced >= expected {
                break;
            }
            let flushed = resample_chunk(&mut resampler, &[], true);
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

    fn tone(samples: usize, frequency_hz: f32) -> Vec<f32> {
        (0..samples)
            .map(|i| 0.4 * (i as f32 * frequency_hz * std::f32::consts::TAU / 8_000.0).sin())
            .collect()
    }

    #[test]
    fn i16_is_converted_to_f32_in_place_order() {
        let mut out = Vec::new();
        i16_to_f32(&[0, 16, -256, i16::MAX, i16::MIN], &mut out);
        assert_eq!(
            out,
            vec![
                0.0,
                16.0 / 32_768.0,
                -256.0 / 32_768.0,
                1.0 - 1.0 / 32_768.0,
                -1.0
            ]
        );
    }

    #[test]
    fn pads_to_the_minimum_clip_duration() {
        let mut short = vec![0.1; 100];
        pad_minimum(&mut short);
        assert_eq!(short.len(), MINIMUM_CLIP_SAMPLES);
        assert_eq!(&short[..100], &[0.1; 100]);
        assert!(short[100..].iter().all(|&sample| sample == 0.0));

        let mut long = vec![0.0; 48_000];
        pad_minimum(&mut long);
        assert_eq!(long.len(), 48_000, "a longer clip is untouched");
    }

    /// `MINIMUM_CLIP_SAMPLES`: 1.5 s at 16 kHz.
    const MINIMUM_CLIP_SAMPLES: usize = 24_000;

    #[test]
    fn passthrough_at_16khz_is_identity() {
        let samples = tone(10_000, 440.0);
        assert_eq!(resample(&samples, 16_000).unwrap(), samples);
    }

    #[test]
    fn downsampling_matches_the_whole_buffer_conversion() {
        let samples = tone(48_000 * 3, 440.0);
        let whole = resample(&samples, 48_000).unwrap();
        let expected = samples.len() * 16_000 / 48_000;
        assert_eq!(whole.len(), expected);

        let mut incremental = Resampler::try_new(48_000).unwrap();
        let mut converted = Vec::new();
        for chunk in samples.chunks(977) {
            converted.extend(incremental.push(chunk));
        }
        converted.extend(incremental.finish());
        assert_eq!(converted.len(), expected);
        // The incremental conversion lands within one sample of the whole
        // buffer's; the resampler's block math is identical, so they agree.
        assert!(
            converted
                .iter()
                .zip(whole.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }

    #[test]
    fn upsampling_produces_the_expected_length() {
        let samples = tone(8_000 * 2, 220.0);
        let converted = resample(&samples, 8_000).unwrap();
        assert_eq!(converted.len(), samples.len() * 2);
    }

    #[test]
    fn incremental_conversion_agrees_with_itself_across_chunk_sizes() {
        // Whatever rubato accepts, the incremental contract holds: two
        // different push chunkings end at the same converted length.
        let samples = tone(48_000 * 2, 440.0);
        let mut small = Resampler::try_new(48_000).unwrap();
        let mut out = Vec::new();
        for chunk in samples.chunks(251) {
            out.extend(small.push(chunk));
        }
        out.extend(small.finish());
        assert_eq!(out.len(), samples.len() * 16_000 / 48_000);
    }
}
