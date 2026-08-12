use std::f64::consts::TAU;
use std::path::Path;

use anyhow::{Result, ensure};
use beat_this::{RtenRuntime, Runtime};
use musical_key_cnn::KeyDetector;
use serde::Deserialize;

#[derive(Deserialize)]
struct PipelineFixture {
    sample_rate: u32,
    sample_count: usize,
    camelot: String,
    probabilities: [f32; 24],
}

fn synthetic_audio(sample_rate: u32, sample_count: usize) -> Vec<f32> {
    (0..sample_count)
        .map(|index| {
            let time = index as f64 / f64::from(sample_rate);
            let envelope = 0.55 + 0.45 * (TAU * 0.37 * time).sin().powi(2);
            let signal = envelope
                * (0.42 * (TAU * 130.8128 * time).sin()
                    + 0.31 * (TAU * 155.5635 * time + 0.2).sin()
                    + 0.23 * (TAU * 195.9977 * time + 0.7).sin()
                    + 0.12 * (TAU * (261.6256 + 0.8 * time) * time).sin());
            signal as f32
        })
        .collect()
}

#[test]
fn rust_pipeline_matches_librosa_and_pytorch() -> Result<()> {
    let fixture: PipelineFixture =
        serde_json::from_str(include_str!("fixtures/synthetic_pipeline.json"))?;
    let model = RtenRuntime.load_model(Path::new("models/keynet.onnx"))?;
    let mut detector = KeyDetector::new(model);
    let estimate = detector.detect(
        &synthetic_audio(fixture.sample_rate, fixture.sample_count),
        fixture.sample_rate,
    )?;
    ensure!(estimate.key.to_string() == fixture.camelot, "key differs");
    let maximum_difference = estimate
        .probabilities
        .iter()
        .zip(fixture.probabilities)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    ensure!(
        maximum_difference < 2e-5,
        "Rust pipeline differs from librosa/PyTorch by {maximum_difference}"
    );
    Ok(())
}
