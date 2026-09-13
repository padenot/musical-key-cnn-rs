use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use beat_this::{Model, RtenRuntime, Runtime};
use clap::{Parser, ValueEnum};
#[cfg(all(target_os = "macos", feature = "coreml"))]
use musical_key_cnn::{CoreMlAcceleration, RustnnCoremlModel};
use musical_key_cnn::{
    KeyDetector, KeyEstimate, KeyPreprocessor, MAX_COREML_FRAMES, PreparedAudio, load_mono,
};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

const DEFAULT_ONNX_MODEL: &str = "models/keynet.onnx";
const DEFAULT_RUSTNN_GRAPH: &str = "models/keynet.json";
const DEFAULT_COREML_MODEL: &str = "models/keynet.mlmodelc";

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "MusicalKeyCNN inference with Core ML and RTen"
)]
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
    let (samples, sample_rate) = load_mono(&cli.audio)
        .with_context(|| format!("could not decode {}", cli.audio.display()))?;
    let prepared = KeyPreprocessor::new().prepare(&samples, sample_rate)?;
    let mut models = load_models(&cli)?;
    let (backend, estimate) = models.detect(&prepared)?;
    info!(backend, "completed MusicalKeyCNN inference");
    print_estimate(&estimate)?;
    Ok(())
}

struct LoadedModels {
    choice: RuntimeChoice,
    coreml: Option<KeyDetector<DynamicModel>>,
    rten: Option<KeyDetector<DynamicModel>>,
}

impl LoadedModels {
    fn detect(&mut self, prepared: &PreparedAudio) -> Result<(&'static str, KeyEstimate)> {
        let frames = prepared.frame_count();
        match self.choice {
            RuntimeChoice::Coreml if frames > MAX_COREML_FRAMES => anyhow::bail!(
                "Core ML accepts at most {MAX_COREML_FRAMES} CQT frames, got {frames}; use --runtime rten"
            ),
            RuntimeChoice::Coreml => {
                let detector = self
                    .coreml
                    .as_mut()
                    .context("Core ML model disappeared before inference")?;
                detector
                    .detect_prepared(prepared)
                    .map(|estimate| ("rustnn-coreml", estimate))
                    .map_err(Into::into)
            }
            RuntimeChoice::Auto if frames <= MAX_COREML_FRAMES && self.coreml.is_some() => {
                let detector = self
                    .coreml
                    .as_mut()
                    .context("Core ML model disappeared before inference")?;
                detector
                    .detect_prepared(prepared)
                    .map(|estimate| ("rustnn-coreml", estimate))
                    .map_err(Into::into)
            }
            RuntimeChoice::Auto if frames > MAX_COREML_FRAMES => {
                warn!(
                    frames,
                    maximum_coreml_frames = MAX_COREML_FRAMES,
                    "track exceeds Core ML time bound; using full-track RTen inference"
                );
                detect_rten(&mut self.rten, prepared)
            }
            RuntimeChoice::Rten | RuntimeChoice::Auto => detect_rten(&mut self.rten, prepared),
        }
    }
}

fn detect_rten(
    detector: &mut Option<KeyDetector<DynamicModel>>,
    prepared: &PreparedAudio,
) -> Result<(&'static str, KeyEstimate)> {
    detector
        .as_mut()
        .context("RTen model is unavailable")?
        .detect_prepared(prepared)
        .map(|estimate| ("rten", estimate))
        .map_err(Into::into)
}

fn load_models(cli: &Cli) -> Result<LoadedModels> {
    let coreml = if cli.runtime != RuntimeChoice::Rten {
        match load_coreml(&cli.graph, &cli.coreml_model) {
            Ok(model) => Some(KeyDetector::new(model)),
            Err(error) if cli.runtime == RuntimeChoice::Auto => {
                warn!(%error, "Core ML model unavailable; using RTen");
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    let rten = if cli.runtime != RuntimeChoice::Coreml {
        Some(KeyDetector::new(load_rten(&cli.model)?))
    } else {
        None
    };
    Ok(LoadedModels {
        choice: cli.runtime,
        coreml,
        rten,
    })
}

fn load_rten(path: &Path) -> Result<DynamicModel> {
    let model = RtenRuntime
        .load_model(path)
        .with_context(|| format!("could not load ONNX model {}", path.display()))?;
    Ok(DynamicModel(Box::new(model)))
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn load_coreml(graph: &Path, compiled_model: &Path) -> Result<DynamicModel> {
    let model = RustnnCoremlModel::load_aot(graph, compiled_model, CoreMlAcceleration::Gpu)
        .with_context(|| {
            format!(
                "could not load RustNN graph {} with Core ML model {}",
                graph.display(),
                compiled_model.display(),
            )
        })?;
    Ok(DynamicModel(Box::new(model)))
}

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn load_coreml(_graph: &Path, _compiled_model: &Path) -> Result<DynamicModel> {
    anyhow::bail!("Core ML support is unavailable in this build")
}

fn print_estimate(estimate: &KeyEstimate) -> Result<()> {
    let stdout = std::io::stdout();
    serde_json::to_writer_pretty(stdout.lock(), estimate).context("could not write result")?;
    println!();
    Ok(())
}
