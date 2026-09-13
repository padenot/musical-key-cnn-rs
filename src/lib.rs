#[cfg(feature = "cli")]
mod audio;
mod key;
mod preprocessor;

#[cfg(all(target_os = "macos", feature = "coreml"))]
use std::path::Path;

#[cfg(feature = "cli")]
pub use audio::load_mono;
#[cfg(all(target_os = "macos", feature = "coreml"))]
pub use beat_this::{CoreMlAcceleration, RustnnCoremlModel};
use beat_this::{Model, Tensor};
pub use beat_this::{RtenRuntime, Runtime};
pub use key::{CamelotKey, KeyMode};
pub use preprocessor::{KeyPreprocessor, PreparedAudio};
use serde::Serialize;
use thiserror::Error;

const CLASS_COUNT: usize = 24;
const INPUT_NAME: &str = "spectrogram";
const OUTPUT_NAME: &str = "logits";

/// Smallest time dimension that survives the model's three 2x2 pooling stages.
pub const MIN_MODEL_FRAMES: usize = 8;
/// Maximum variable time dimension supported by the compiled Core ML graph.
pub const MAX_COREML_FRAMES: usize = 4_096;

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
    preprocessor: KeyPreprocessor,
}

impl<M: Model> KeyDetector<M> {
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            preprocessor: KeyPreprocessor::new(),
        }
    }

    pub fn detect(&mut self, mono: &[f32], sample_rate: u32) -> Result<KeyEstimate> {
        let prepared = self.preprocessor.prepare(mono, sample_rate)?;
        self.detect_prepared(&prepared)
    }

    pub fn detect_prepared(&mut self, prepared: &PreparedAudio) -> Result<KeyEstimate> {
        let logits = infer_logits(&mut self.model, &prepared.spectrogram)?;
        let probabilities = softmax(logits)?;
        estimate_from_probabilities(probabilities)
    }

    /// Detect several independently-sized prepared tracks in one runtime batch.
    pub fn detect_prepared_batch(
        &mut self,
        prepared: &[&PreparedAudio],
    ) -> Result<Vec<KeyEstimate>> {
        if prepared.is_empty() {
            return Err(Error::InvalidAudio(
                "key inference batch is empty".to_owned(),
            ));
        }
        let inputs = prepared
            .iter()
            .map(|audio| [(INPUT_NAME, &audio.spectrogram)])
            .collect::<Vec<_>>();
        let batches = inputs
            .iter()
            .map(|inputs| inputs.as_slice())
            .collect::<Vec<_>>();
        let outputs = self.model.run_batch(&batches).map_err(Error::Model)?;
        if outputs.len() != prepared.len() {
            return Err(Error::InvalidModelOutput(format!(
                "model returned {} key batches for {} prepared tracks",
                outputs.len(),
                prepared.len()
            )));
        }
        outputs
            .into_iter()
            .map(|outputs| {
                let logits = logits_from_outputs(outputs)?;
                let probabilities = softmax(logits)?;
                estimate_from_probabilities(probabilities)
            })
            .collect()
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
        acceleration: CoreMlAcceleration,
    ) -> Result<Self> {
        let model = RustnnCoremlModel::load_aot(graph, compiled_model, acceleration)
            .map_err(Error::Model)?;
        Ok(Self::new(model))
    }
}

fn infer_logits<M: Model>(model: &mut M, spectrogram: &Tensor) -> Result<[f32; CLASS_COUNT]> {
    let outputs = model
        .run(&[(INPUT_NAME, spectrogram)])
        .map_err(Error::Model)?;
    logits_from_outputs(outputs)
}

fn logits_from_outputs(
    mut outputs: std::collections::HashMap<String, Tensor>,
) -> Result<[f32; CLASS_COUNT]> {
    let output = outputs.remove(OUTPUT_NAME).ok_or_else(|| {
        Error::InvalidModelOutput(format!("model did not return {OUTPUT_NAME:?}"))
    })?;
    if output.shape != [1, CLASS_COUNT]
        || output.data.len() != CLASS_COUNT
        || output.data.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidModelOutput(format!(
            "expected [1, {CLASS_COUNT}] finite logits, got {} values with shape {:?}",
            output.data.len(),
            output.shape,
        )));
    }
    output.data.try_into().map_err(|values: Vec<f32>| {
        Error::InvalidModelOutput(format!(
            "expected {CLASS_COUNT} logits, got {}",
            values.len()
        ))
    })
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
        let first = probabilities
            .first()
            .ok_or("softmax returned no probabilities")?;
        assert_abs_diff_eq!(*first, 1.0 / 24.0, epsilon = 1e-6);
        Ok(())
    }
}
