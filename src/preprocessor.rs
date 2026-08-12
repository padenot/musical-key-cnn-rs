use beat_this::Tensor;
use rosa::{CqtParams, complex_magnitude, cqt, resample};

use crate::{Error, Result};

const MODEL_SAMPLE_RATE: u32 = 44_100;
const HOP_LENGTH: usize = 8_820;
const FREQUENCY_BINS: usize = 105;
const BINS_PER_OCTAVE: usize = 24;
const MINIMUM_FREQUENCY_HZ: f64 = 65.0;
const MODEL_FRAMES: usize = 512;

pub(crate) struct PreparedChunk {
    pub tensor: Tensor,
    pub source_frames: usize,
}

pub(crate) struct SpectrogramPreprocessor;

impl SpectrogramPreprocessor {
    pub fn prepare(&self, mono: &[f32], sample_rate: u32) -> Result<Vec<PreparedChunk>> {
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
        if magnitude.rows() != FREQUENCY_BINS || magnitude.cols() == 0 {
            return Err(Error::InvalidAudio(format!(
                "CQT returned shape {:?}, expected {FREQUENCY_BINS} non-empty rows",
                magnitude.shape(),
            )));
        }
        let log_magnitude = magnitude.map(f64::ln_1p);
        chunk_spectrogram(
            log_magnitude.as_slice(),
            log_magnitude.rows(),
            log_magnitude.cols(),
        )
    }
}

fn chunk_spectrogram(values: &[f64], rows: usize, columns: usize) -> Result<Vec<PreparedChunk>> {
    let expected_values = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::InvalidAudio("CQT shape overflow".to_owned()))?;
    if rows != FREQUENCY_BINS || columns == 0 || values.len() != expected_values {
        return Err(Error::InvalidAudio(format!(
            "invalid CQT buffer: {rows}x{columns} for {} values",
            values.len(),
        )));
    }
    let chunk_count = columns.div_ceil(MODEL_FRAMES);
    let mut chunks = Vec::with_capacity(chunk_count);
    for chunk_index in 0..chunk_count {
        let start = chunk_index
            .checked_mul(columns)
            .and_then(|value| value.checked_div(chunk_count))
            .ok_or_else(|| Error::InvalidAudio("CQT chunk start overflow".to_owned()))?;
        let end = (chunk_index + 1)
            .checked_mul(columns)
            .and_then(|value| value.checked_div(chunk_count))
            .ok_or_else(|| Error::InvalidAudio("CQT chunk end overflow".to_owned()))?;
        let source_frames = end.saturating_sub(start);
        if source_frames == 0 {
            return Err(Error::InvalidAudio(
                "CQT chunk contains no source frames".to_owned(),
            ));
        }
        let value_count = FREQUENCY_BINS
            .checked_mul(MODEL_FRAMES)
            .ok_or_else(|| Error::InvalidAudio("model input size overflow".to_owned()))?;
        let mut data = Vec::with_capacity(value_count);
        for row in values.chunks_exact(columns) {
            let source = row
                .get(start..end)
                .ok_or_else(|| Error::InvalidAudio("CQT chunk is outside its row".to_owned()))?;
            data.extend(
                source
                    .iter()
                    .cycle()
                    .take(MODEL_FRAMES)
                    .map(|&value| value as f32),
            );
        }
        chunks.push(PreparedChunk {
            tensor: Tensor {
                shape: vec![1, 1, FREQUENCY_BINS, MODEL_FRAMES],
                data,
            },
            source_frames,
        });
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::{FREQUENCY_BINS, MODEL_FRAMES, chunk_spectrogram};

    #[test]
    fn partitions_long_spectrograms_into_balanced_chunks() -> Result<(), Box<dyn std::error::Error>>
    {
        let columns = MODEL_FRAMES + 188;
        let values = vec![1.0; FREQUENCY_BINS * columns];
        let chunks = chunk_spectrogram(&values, FREQUENCY_BINS, columns)?;
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.source_frames)
                .collect::<Vec<_>>(),
            [350, 350],
        );
        assert!(chunks.iter().all(|chunk| {
            chunk.tensor.shape == [1, 1, FREQUENCY_BINS, MODEL_FRAMES]
                && chunk.tensor.data.len() == FREQUENCY_BINS * MODEL_FRAMES
        }));
        Ok(())
    }
}
