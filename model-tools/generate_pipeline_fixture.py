from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import torch

from export_model import load_model
from reference_predict import camelot_class, chunked_logits


LOGGER = logging.getLogger("musical_key_cnn.pipeline_fixture")
SAMPLE_RATE = 44_100
DURATION_SECONDS = 24


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic librosa/PyTorch pipeline fixture."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def synthetic_audio() -> np.ndarray:
    sample_count = SAMPLE_RATE * DURATION_SECONDS
    time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 0.37 * time) ** 2
    signal = envelope * (
        0.42 * np.sin(2 * np.pi * 130.8128 * time)
        + 0.31 * np.sin(2 * np.pi * 155.5635 * time + 0.2)
        + 0.23 * np.sin(2 * np.pi * 195.9977 * time + 0.7)
        + 0.12 * np.sin(2 * np.pi * (261.6256 + 0.8 * time) * time)
    )
    return signal.astype(np.float32)


def main() -> int:
    arguments = parse_arguments()
    spectrogram = np.log1p(
        np.abs(
            librosa.cqt(
                synthetic_audio(),
                sr=SAMPLE_RATE,
                hop_length=8_820,
                n_bins=105,
                bins_per_octave=24,
                fmin=65,
            )
        )
    ).astype(np.float32)
    model = load_model(arguments.checkpoint)
    with torch.inference_mode():
        logits = chunked_logits(model, spectrogram)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
    payload = {
        "sample_rate": SAMPLE_RATE,
        "sample_count": SAMPLE_RATE * DURATION_SECONDS,
        "frames": int(spectrogram.shape[1]),
        **camelot_class(probabilities),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("wrote %s", arguments.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("pipeline fixture generation failed")
        raise SystemExit(1)
