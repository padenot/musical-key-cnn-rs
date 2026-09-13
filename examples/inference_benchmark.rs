use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use beat_this::{CoreMlAcceleration, RtenRuntime, Runtime};
use clap::{Parser, ValueEnum};
use musical_key_cnn::{KeyDetector, KeyPreprocessor, RustnnCoremlModel};

const DEFAULT_ONNX_MODEL: &str = "models/keynet.onnx";
const DEFAULT_RUSTNN_GRAPH: &str = "models/keynet.json";
const DEFAULT_COREML_MODEL: &str = "models/keynet.mlmodelc";

#[derive(Debug, Parser)]
#[command(about = "Release benchmark for MusicalKeyCNN preprocessing and inference")]
struct Cli {
    #[arg(required_unless_present = "fixture", conflicts_with = "fixture")]
    audio: Option<PathBuf>,
    /// Benchmark the canonical 24-second pipeline fixture.
    #[arg(long)]
    fixture: bool,
    #[arg(long, value_enum, default_value_t = Backend::All)]
    backend: Backend,
    #[arg(long, default_value_t = 11)]
    iterations: usize,
    #[arg(long, default_value = DEFAULT_ONNX_MODEL)]
    model: PathBuf,
    #[arg(long, default_value = DEFAULT_RUSTNN_GRAPH)]
    graph: PathBuf,
    #[arg(long, default_value = DEFAULT_COREML_MODEL)]
    coreml_model: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    All,
    Rten,
    Gpu,
    NeuralEngine,
}

fn main() -> Result<()> {
    let _ = env_logger::Builder::from_default_env().try_init();
    let cli = Cli::parse();
    ensure!(
        cli.iterations >= 10,
        "benchmark needs at least 10 iterations"
    );

    let decode_started = Instant::now();
    let (source, samples, sample_rate) = load_audio(&cli)?;
    let decode_elapsed = decode_started.elapsed();

    let prepare_started = Instant::now();
    let prepared = KeyPreprocessor::new().prepare(&samples, sample_rate)?;
    let prepare_elapsed = prepare_started.elapsed();
    println!(
        "audio={} sample_rate={} frames={} decode_ms={:.3} cqt_ms={:.3}",
        source,
        sample_rate,
        prepared.frame_count(),
        milliseconds(decode_elapsed),
        milliseconds(prepare_elapsed),
    );

    if matches!(cli.backend, Backend::All | Backend::Rten) {
        let load_started = Instant::now();
        let model = RtenRuntime
            .load_model(&cli.model)
            .with_context(|| format!("could not load RTen model {}", cli.model.display()))?;
        benchmark(
            "rten/cpu",
            load_started.elapsed(),
            KeyDetector::new(model),
            &prepared,
            cli.iterations,
        )?;
    }
    if matches!(cli.backend, Backend::All | Backend::Gpu) {
        benchmark_coreml(
            "coreml/cpu+gpu",
            CoreMlAcceleration::Gpu,
            &cli.graph,
            &cli.coreml_model,
            &prepared,
            cli.iterations,
        )?;
    }
    if matches!(cli.backend, Backend::All | Backend::NeuralEngine) {
        benchmark_coreml(
            "coreml/cpu+ane",
            CoreMlAcceleration::NeuralEngine,
            &cli.graph,
            &cli.coreml_model,
            &prepared,
            cli.iterations,
        )?;
    }
    Ok(())
}

fn load_audio(cli: &Cli) -> Result<(String, Vec<f32>, u32)> {
    if cli.fixture {
        let sample_rate = 44_100_u32;
        let sample_count = 1_058_400_usize;
        let samples = (0..sample_count)
            .map(|index| {
                let time = index as f64 / f64::from(sample_rate);
                let envelope = 0.55 + 0.45 * (std::f64::consts::TAU * 0.37 * time).sin().powi(2);
                (envelope
                    * (0.42 * (std::f64::consts::TAU * 130.8128 * time).sin()
                        + 0.31 * (std::f64::consts::TAU * 155.5635 * time + 0.2).sin()
                        + 0.23 * (std::f64::consts::TAU * 195.9977 * time + 0.7).sin()
                        + 0.12 * (std::f64::consts::TAU * (261.6256 + 0.8 * time) * time).sin()))
                    as f32
            })
            .collect();
        return Ok(("canonical-fixture".to_owned(), samples, sample_rate));
    }
    let path = cli.audio.as_deref().context("audio path is required")?;
    let (samples, sample_rate) = musical_key_cnn::load_mono(path)
        .with_context(|| format!("could not decode {}", path.display()))?;
    Ok((path.display().to_string(), samples, sample_rate))
}

fn benchmark_coreml(
    label: &str,
    acceleration: CoreMlAcceleration,
    graph: &Path,
    compiled_model: &Path,
    prepared: &musical_key_cnn::PreparedAudio,
    iterations: usize,
) -> Result<()> {
    let load_started = Instant::now();
    let model = RustnnCoremlModel::load_aot(graph, compiled_model, acceleration)
        .with_context(|| format!("could not load {label}"))?;
    benchmark(
        label,
        load_started.elapsed(),
        KeyDetector::new(model),
        prepared,
        iterations,
    )
}

fn benchmark<M: beat_this::Model>(
    label: &str,
    load_elapsed: Duration,
    mut detector: KeyDetector<M>,
    prepared: &musical_key_cnn::PreparedAudio,
    iterations: usize,
) -> Result<()> {
    let warmup = detector.detect_prepared(prepared)?;
    let mut timings = Vec::with_capacity(iterations);
    let mut last_estimate = warmup;
    for _ in 0..iterations {
        let started = Instant::now();
        last_estimate = detector.detect_prepared(prepared)?;
        timings.push(started.elapsed());
    }
    timings.sort_unstable();
    let minimum = timings
        .first()
        .copied()
        .context("benchmark produced no timings")?;
    let median = timings
        .get(timings.len() / 2)
        .copied()
        .context("benchmark produced no median")?;
    println!(
        "backend={label} load_ms={:.3} iterations={iterations} min_ms={:.3} median_ms={:.3} key={} probability={:.6}",
        milliseconds(load_elapsed),
        milliseconds(minimum),
        milliseconds(median),
        last_estimate.key,
        last_estimate.probability,
    );
    Ok(())
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
