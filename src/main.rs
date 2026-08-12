use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
#[cfg(not(target_os = "macos"))]
use anyhow::bail;
use beat_this::{Model, RtenRuntime, Runtime};
use clap::{Parser, ValueEnum};
use musical_key_cnn::{KeyDetector, KeyEstimate};
#[cfg(target_os = "macos")]
use musical_key_cnn::RustnnCoremlModel;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

const DEFAULT_ONNX_MODEL: &str = "models/keynet.onnx";
const DEFAULT_RUSTNN_GRAPH: &str = "models/keynet.json";
const DEFAULT_COREML_MODEL: &str = "models/keynet.mlmodelc";

#[derive(Debug, Parser)]
#[command(author, version, about = "MusicalKeyCNN inference with Core ML and RTen")]
struct Cli {
    audio: PathBuf,
    #[arg(long, value_enum, default_value_t = RuntimeChoice::Auto)]
    runtime: RuntimeChoice,
    #[arg(long, default_value = DEFAULT_ONNX_MODEL)]
    model: PathBuf,
    #[arg(long, default_value = DEFAULT_RUSTNN_GRAPH)]
    graph: PathBuf,
    #[arg(long, default_value = DEFAULT_COREML_MODEL)]
    coreml_model: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum RuntimeChoice {
    Auto,
    Coreml,
    Rten,
}

struct DynamicModel(Box<dyn Model>);

impl Model for DynamicModel {
    fn max_batch_size(&self) -> usize {
        self.0.max_batch_size()
    }

    fn run(
        &mut self,
        inputs: &[(&str, &beat_this::Tensor)],
    ) -> anyhow::Result<std::collections::HashMap<String, beat_this::Tensor>> {
        self.0.run(inputs)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    if let Err(source) = run(cli) {
        error!(error = %source, "key analysis failed");
        return Err(source);
    }
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    let audio_path = cli
        .audio
        .to_str()
        .context("audio path is not valid UTF-8")?;
    let (samples, sample_rate) = rosa::load(audio_path, None, true)
        .with_context(|| format!("could not decode {}", cli.audio.display()))?;
    let samples = samples.into_iter().map(|sample| sample as f32).collect::<Vec<_>>();
    let (backend, model) = load_model(&cli)?;
    info!(backend, "loaded MusicalKeyCNN model");
    let mut detector = KeyDetector::new(model);
    let estimate = detector.detect(&samples, sample_rate)?;
    print_estimate(&estimate)?;
    Ok(())
}

fn load_model(cli: &Cli) -> Result<(&'static str, DynamicModel)> {
    match cli.runtime {
        RuntimeChoice::Rten => load_rten(&cli.model),
        RuntimeChoice::Coreml => load_coreml(&cli.graph, &cli.coreml_model),
        RuntimeChoice::Auto => {
            #[cfg(target_os = "macos")]
            if cli.graph.is_file() && cli.coreml_model.is_dir() {
                return load_coreml(&cli.graph, &cli.coreml_model);
            }
            load_rten(&cli.model)
        }
    }
}

fn load_rten(path: &Path) -> Result<(&'static str, DynamicModel)> {
    let model = RtenRuntime
        .load_model(path)
        .with_context(|| format!("could not load ONNX model {}", path.display()))?;
    Ok(("rten", DynamicModel(Box::new(model))))
}

#[cfg(target_os = "macos")]
fn load_coreml(graph: &Path, compiled_model: &Path) -> Result<(&'static str, DynamicModel)> {
    let model = RustnnCoremlModel::load_aot_batched(graph, compiled_model, 8).with_context(|| {
        format!(
            "could not load RustNN graph {} with Core ML model {}",
            graph.display(),
            compiled_model.display(),
        )
    })?;
    Ok(("rustnn-coreml", DynamicModel(Box::new(model))))
}

#[cfg(not(target_os = "macos"))]
fn load_coreml(_graph: &Path, _compiled_model: &Path) -> Result<(&'static str, DynamicModel)> {
    bail!("Core ML is only available on macOS")
}

fn print_estimate(estimate: &KeyEstimate) -> Result<()> {
    let stdout = std::io::stdout();
    serde_json::to_writer_pretty(stdout.lock(), estimate).context("could not write result")?;
    println!();
    Ok(())
}
