use std::path::Path;

use anyhow::{Context, Result, ensure};
#[cfg(all(target_os = "macos", feature = "coreml"))]
use beat_this::RustnnCoremlModel;
use beat_this::{Model, RtenRuntime, Runtime, Tensor};

fn input() -> Tensor {
    let shape = vec![1, 1, 105, 512];
    let data = (0..(105 * 512))
        .map(|index| ((index % 257) as f32 - 128.0) / 256.0)
        .collect();
    Tensor { shape, data }
}

#[test]
#[cfg(all(target_os = "macos", feature = "coreml"))]
fn coreml_matches_rten() -> Result<()> {
    let input = input();
    let mut rten = RtenRuntime.load_model(Path::new("models/keynet.onnx"))?;
    let mut coreml = RustnnCoremlModel::load_aot(
        Path::new("models/keynet.json"),
        Path::new("models/keynet.mlmodelc"),
    )?;
    let expected = rten
        .run(&[("spectrogram", &input)])?
        .remove("logits")
        .context("RTen returned no logits")?;
    let actual = coreml
        .run(&[("spectrogram", &input)])?
        .remove("logits")
        .context("Core ML returned no logits")?;
    ensure!(expected.shape == actual.shape, "output shapes differ");
    let maximum_difference = expected
        .data
        .iter()
        .zip(&actual.data)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    ensure!(
        maximum_difference < 1e-3,
        "Core ML differs from RTen by {maximum_difference}"
    );
    Ok(())
}
