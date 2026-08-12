use beat_this::Tensor;
use rosa::{CqtParams, complex_magnitude, cqt, resample};

use crate::{Error, MIN_MODEL_FRAMES, Result};

const MODEL_SAMPLE_RATE: u32 = 44_100;
const HOP_LENGTH: usize = 8_820;
const FREQUENCY_BINS: usize = 105;
const BINS_PER_OCTAVE: usize = 24;
const MINIMUM_FREQUENCY_HZ: f64 = 65.0;
/// Opaque, model-ready MusicalKeyCNN input.
///
/// Preparing audio is CPU-intensive and can safely run in parallel before the
/// prepared input is sent to a shared inference worker.
pub struct PreparedAudio {
    pub(crate) spectrogram: Tensor,
}

impl PreparedAudio {
    /// Number of time frames in the complete CQT spectrogram.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        match self.spectrogram.shape.last() {
            Some(frames) => *frames,
            None => 0,
        }
    }
}

/// Stateless MusicalKeyCNN CQT preprocessor.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyPreprocessor;

impl KeyPreprocessor {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub fn prepare(&self, mono: &[f32], sample_rate: u32) -> Result<PreparedAudio> {
        if sample_rate == 0 {
            return Err(Error::InvalidAudio("sample rate is zero".to_owned()));
        }
        if mono.is_empty() {
            return Err(Error::InvalidAudio("PCM is empty".to_owned()));
        }
        if mono.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidAudio(
                "PCM contains non-finite samples".to_owned(),
            ));
        }
        let mut samples = mono
            .iter()
            .map(|&sample| f64::from(sample))
            .collect::<Vec<_>>();
        if sample_rate != MODEL_SAMPLE_RATE {
            samples = resample(&samples, sample_rate, MODEL_SAMPLE_RATE)
                .map_err(|source| Error::InvalidAudio(format!("resampling failed: {source}")))?;
        }
        if samples.len() < HOP_LENGTH {
            samples.resize(HOP_LENGTH, 0.0);
        }
        let params = CqtParams {
            sr: f64::from(MODEL_SAMPLE_RATE),
            hop_length: HOP_LENGTH,
            fmin: MINIMUM_FREQUENCY_HZ,
            n_bins: FREQUENCY_BINS,
            bins_per_octave: BINS_PER_OCTAVE,
            ..CqtParams::default()
        };
        let (real, imaginary) = cqt(&samples, &params);
        let magnitude = complex_magnitude(&real, &imaginary);
        if magnitude.rows() != FREQUENCY_BINS || magnitude.cols() < MIN_MODEL_FRAMES {
            return Err(Error::InvalidAudio(format!(
                "CQT returned shape {:?}, expected {FREQUENCY_BINS} rows and at least {MIN_MODEL_FRAMES} time frames",
                magnitude.shape(),
            )));
        }
        let log_magnitude = magnitude.map(f64::ln_1p);
        spectrogram_tensor(
            log_magnitude.as_slice(),
            log_magnitude.rows(),
            log_magnitude.cols(),
        )
        .map(|spectrogram| PreparedAudio { spectrogram })
    }
}

fn spectrogram_tensor(values: &[f64], rows: usize, columns: usize) -> Result<Tensor> {
    let expected_values = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::InvalidAudio("CQT shape overflow".to_owned()))?;
    if rows != FREQUENCY_BINS || columns < MIN_MODEL_FRAMES || values.len() != expected_values {
        return Err(Error::InvalidAudio(format!(
            "invalid CQT buffer: {rows}x{columns} for {} values",
            values.len(),
        )));
    }
    Ok(Tensor {
        shape: vec![1, 1, FREQUENCY_BINS, columns],
        data: values.iter().map(|&value| value as f32).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::{FREQUENCY_BINS, spectrogram_tensor};

    #[test]
    fn preserves_complete_variable_width_spectrogram() -> Result<(), Box<dyn std::error::Error>> {
        let columns = 700;
        let values = (0..FREQUENCY_BINS * columns)
            .map(|value| value as f64)
            .collect::<Vec<_>>();
        let tensor = spectrogram_tensor(&values, FREQUENCY_BINS, columns)?;
        assert_eq!(tensor.shape, [1, 1, FREQUENCY_BINS, columns]);
        assert_eq!(tensor.data.len(), values.len());
        assert_eq!(tensor.data.get(699), Some(&699.0));
        assert_eq!(tensor.data.get(700), Some(&700.0));
        Ok(())
    }
}
