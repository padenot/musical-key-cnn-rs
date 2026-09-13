# MusicalKeyCNN Rust

A Rust inference port of [a1ex90/MusicalKeyCNN](https://github.com/a1ex90/MusicalKeyCNN),
using its trained `keynet.pt` weights and 24-class Camelot output. The model is MIT-licensed.

The library accepts decoded mono PCM, computes the model's 105-bin log-magnitude CQT, and runs
the complete variable-width spectrogram. On macOS, the CLI prefers the ahead-of-time RustNN/Core
ML model; the same ONNX model runs through RTen as the portable reference backend. The Core ML
graph accepts 8 through 4,096 CQT frames (roughly 1.6 seconds through 13 minutes 39 seconds);
RTen's symbolic ONNX input has the same minimum and no upper bound.

The fixed model preprocessor lives in this crate and is checked against frozen librosa/PyTorch
fixtures at 44.1 and 48 kHz. Sample-rate conversion uses Rubato's offline FFT resampler. Model
regeneration expects sibling `rustnn` and `onnx2webnn` checkouts under `~/src/repositories`.

## Models

The reproducible chain is:

```text
upstream keynet.pt -> models/keynet.onnx -> models/keynet.json -> models/keynet.mlmodelc
```

Generate the ONNX source model with `uv`:

```sh
uv sync --project model-tools
uv run --project model-tools model-tools/export_model.py \
  ../MusicalKeyCNN/checkpoints/keynet.pt \
  models/keynet.onnx
```

Lower ONNX to a backend-independent RustNN graph:

```sh
cargo run --release --no-default-features \
  --manifest-path ../onnx2webnn/Cargo.toml -- \
  convert --input models/keynet.onnx --output models/keynet.json --optimize \
  --experimental-dynamic-inputs
```

Compile the Core ML model ahead of time:

```sh
cargo run --release --manifest-path ../rustnn/Cargo.toml -- \
  models/keynet.json \
  --convert coreml \
  --convert-output models/keynet.mlpackage \
  --run-coreml \
  --coreml-compiled-output models/keynet.mlmodelc
```

Applications load the committed `.mlmodelc`; they do not convert or compile models at startup.

## CLI

```sh
cargo run --release -- path/to/track.mp3
cargo run --release -- path/to/track.mp3 --runtime rten
```

The default `auto` runtime uses Core ML when both RustNN assets are present and otherwise falls
back to RTen. Pass model paths explicitly when running outside the repository.

## Library

```rust,no_run
use musical_key_cnn::{KeyDetector, RtenRuntime, Runtime};
use std::path::Path;

let model = RtenRuntime.load_model(Path::new("models/keynet.onnx"))?;
let mut detector = KeyDetector::new(model);
let estimate = detector.detect(&mono_pcm, sample_rate)?;
println!("{}", estimate.key);
# Ok::<(), musical_key_cnn::Error>(())
```

`KeyDetector` is synchronous by design. Run it on an analysis worker; neither CQT preprocessing
nor inference belongs on the GUI or real-time audio thread.

## Verification

```sh
cargo test --release --all-targets --all-features
uv run --project model-tools model-tools/export_debug_model.py \
  models/keynet.onnx models/keynet-debug.onnx
cargo test --release --test intermediate_parity -- --ignored --nocapture
```

The first command checks the full Rust CQT + RTen result against deterministic librosa + PyTorch
fixtures, including 48 kHz input resampling. Runtime parity covers variable-width batches. The
ignored diagnostic exposes every neural-network stage and compares RTen to Core ML.
