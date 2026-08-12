mod key;
mod preprocessor;

#[cfg(all(target_os = "macos", feature = "coreml"))]
use std::path::Path;

#[cfg(all(target_os = "macos", feature = "coreml"))]
pub use beat_this::RustnnCoremlModel;
use beat_this::{Model, Tensor};
pub use beat_this::{RtenRuntime, Runtime};
pub use key::{CamelotKey, KeyMode};
use preprocessor::{PreparedChunk, SpectrogramPreprocessor};
use serde::Serialize;
use thiserror::Error;

const CLASS_COUNT: usize = 24;
const INPUT_NAME: &str = "spectrogram";
const OUTPUT_NAME: &str = "logits";

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid audio: {0}")]
    InvalidAudio(String),
    #[error("model error: {0}")]
    Model(#[source] anyhow::Error),
    #[error("invalid model output: {0}")]
    InvalidModelOutput(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize)]
pub struct KeyEstimate {
    pub key: CamelotKey,
    pub probability: f32,
    pub margin: f32,
    pub probabilities: [f32; CLASS_COUNT],
}

pub struct KeyDetector<M> {
    model: M,
    preprocessor: SpectrogramPreprocessor,
}

impl<M: Model> KeyDetector<M> {
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            preprocessor: SpectrogramPreprocessor,
        }
    }

    pub fn detect(&mut self, mono: &[f32], sample_rate: u32) -> Result<KeyEstimate> {
        let chunks = self.preprocessor.prepare(mono, sample_rate)?;
        let logits = infer_weighted_logits(&mut self.model, &chunks)?;
        let probabilities = softmax(logits)?;
        estimate_from_probabilities(probabilities)
    }

    #[must_use]
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
impl KeyDetector<RustnnCoremlModel> {
    pub fn from_coreml_assets(
        graph: &Path,
        compiled_model: &Path,
        native_batch_limit: usize,
    ) -> Result<Self> {
        let model = RustnnCoremlModel::load_aot_batched(graph, compiled_model, native_batch_limit)
            .map_err(Error::Model)?;
        Ok(Self::new(model))
    }
}

fn infer_weighted_logits<M: Model>(
    model: &mut M,
    chunks: &[PreparedChunk],
) -> Result<[f32; CLASS_COUNT]> {
    let batch_limit = model.max_batch_size();
    if batch_limit == 0 {
        return Err(Error::InvalidModelOutput(
            "model reports a zero batch limit".to_owned(),
        ));
    }
    let mut sums = [0.0_f64; CLASS_COUNT];
    let mut total_weight = 0usize;
    for batch in chunks.chunks(batch_limit) {
        let tensor = batch_tensor(batch)?;
        let mut outputs = model.run(&[(INPUT_NAME, &tensor)]).map_err(Error::Model)?;
        let output = outputs.remove(OUTPUT_NAME).ok_or_else(|| {
            Error::InvalidModelOutput(format!("model did not return {OUTPUT_NAME:?}"))
        })?;
        let expected = batch
            .len()
            .checked_mul(CLASS_COUNT)
            .ok_or_else(|| Error::InvalidModelOutput("model output size overflow".to_owned()))?;
        if output.data.len() != expected || output.data.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidModelOutput(format!(
                "expected {expected} finite logits, got {} with shape {:?}",
                output.data.len(),
                output.shape,
            )));
        }
        for (logits, chunk) in output.data.chunks_exact(CLASS_COUNT).zip(batch) {
            let weight = chunk.source_frames;
            total_weight = total_weight.checked_add(weight).ok_or_else(|| {
                Error::InvalidModelOutput("spectrogram weight overflow".to_owned())
            })?;
            for (sum, &logit) in sums.iter_mut().zip(logits) {
                *sum += f64::from(logit) * weight as f64;
            }
        }
    }
    if total_weight == 0 {
        return Err(Error::InvalidModelOutput(
            "model received no weighted spectrogram frames".to_owned(),
        ));
    }
    Ok(sums.map(|sum| (sum / total_weight as f64) as f32))
}

fn batch_tensor(chunks: &[PreparedChunk]) -> Result<Tensor> {
    let Some(first) = chunks.first() else {
        return Err(Error::InvalidModelOutput(
            "cannot infer an empty spectrogram batch".to_owned(),
        ));
    };
    let values_per_chunk = first.tensor.data.len();
    let capacity = values_per_chunk
        .checked_mul(chunks.len())
        .ok_or_else(|| Error::InvalidModelOutput("spectrogram batch size overflow".to_owned()))?;
    let mut data = Vec::with_capacity(capacity);
    for chunk in chunks {
        if chunk.tensor.shape != first.tensor.shape || chunk.tensor.data.len() != values_per_chunk {
            return Err(Error::InvalidModelOutput(
                "spectrogram chunks have inconsistent shapes".to_owned(),
            ));
        }
        data.extend_from_slice(&chunk.tensor.data);
    }
    let mut shape = first.tensor.shape.clone();
    let batch_dimension = shape.first_mut().ok_or_else(|| {
        Error::InvalidModelOutput("spectrogram tensor has no batch dimension".to_owned())
    })?;
    *batch_dimension = chunks.len();
    Ok(Tensor { shape, data })
}

fn softmax(logits: [f32; CLASS_COUNT]) -> Result<[f32; CLASS_COUNT]> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = logits.map(|value| (value - maximum).exp());
    let total = exponentials.iter().sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::InvalidModelOutput(
            "softmax normalization is not finite".to_owned(),
        ));
    }
    Ok(exponentials.map(|value| value / total))
}

fn estimate_from_probabilities(probabilities: [f32; CLASS_COUNT]) -> Result<KeyEstimate> {
    let mut ranked = probabilities
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    let (best_class, probability) = ranked.first().copied().ok_or_else(|| {
        Error::InvalidModelOutput("model returned no key probabilities".to_owned())
    })?;
    let runner_up = ranked.get(1).map_or(0.0, |candidate| candidate.1);
    let key = CamelotKey::from_class(best_class)?;
    Ok(KeyEstimate {
        key,
        probability,
        margin: (probability - runner_up).max(0.0),
        probabilities,
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::softmax;

    #[test]
    fn softmax_is_normalized() -> Result<(), Box<dyn std::error::Error>> {
        let probabilities = softmax([0.0; 24])?;
        assert_abs_diff_eq!(probabilities.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(probabilities[0], 1.0 / 24.0, epsilon = 1e-6);
        Ok(())
    }
}
