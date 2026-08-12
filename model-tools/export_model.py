from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator


FREQUENCY_BINS = 105
EXPORT_FRAMES = 512
VALIDATION_FRAMES = (8, 121, EXPORT_FRAMES, 1_501)
LOGGER = logging.getLogger("musical_key_cnn.export")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the upstream MusicalKeyCNN checkpoint to variable-width ONNX."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def load_upstream_model_module(checkpoint: Path) -> ModuleType:
    module_path = checkpoint.parent.parent / "model.py"
    spec = importlib.util.spec_from_file_location("musical_key_cnn_upstream", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load upstream model module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(checkpoint: Path) -> torch.nn.Module:
    module = load_upstream_model_module(checkpoint)
    model_type = getattr(module, "KeyNet", None)
    if model_type is None:
        raise RuntimeError("upstream model.py has no KeyNet class")
    model = model_type(num_classes=24, in_channels=1, Nf=20)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def main() -> int:
    arguments = parse_arguments()
    model = load_model(arguments.checkpoint)
    random = np.random.default_rng(0x4B4559)
    example = random.normal(
        size=(1, 1, FREQUENCY_BINS, EXPORT_FRAMES)
    ).astype(np.float32)
    tensor = torch.from_numpy(example)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        tensor,
        arguments.output,
        input_names=["spectrogram"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={"spectrogram": {3: "sequence_length"}},
    )

    exported = onnx.load(arguments.output)
    onnx.checker.check_model(exported)
    evaluator = ReferenceEvaluator(exported)
    maximum_error = 0.0
    with torch.inference_mode():
        for frames in VALIDATION_FRAMES:
            sample = random.normal(
                size=(1, 1, FREQUENCY_BINS, frames)
            ).astype(np.float32)
            expected = model(torch.from_numpy(sample)).numpy()
            actual = evaluator.run(None, {"spectrogram": sample})[0]
            error = float(np.max(np.abs(actual - expected)))
            maximum_error = max(maximum_error, error)
            if not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                raise RuntimeError(
                    "ONNX output differs from PyTorch at "
                    f"{frames} frames; maximum error {error:.8f}"
                )
    LOGGER.info(
        "exported %s (%d bytes), maximum PyTorch error %.8f",
        arguments.output,
        arguments.output.stat().st_size,
        maximum_error,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("model export failed")
        raise SystemExit(1)
