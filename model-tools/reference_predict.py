from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from export_model import load_model

MODEL_FRAMES = 512


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the upstream MusicalKeyCNN preprocessing and PyTorch model."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    return parser.parse_args()


def camelot_class(probabilities: np.ndarray) -> dict[str, object]:
    predicted = int(np.argmax(probabilities))
    return {
        "class": predicted,
        "camelot": f"{predicted % 12 + 1}{'A' if predicted < 12 else 'B'}",
        "probability": float(probabilities[predicted]),
        "probabilities": probabilities.tolist(),
    }


def chunked_logits(model: torch.nn.Module, spectrogram: np.ndarray) -> np.ndarray:
    frame_count = spectrogram.shape[1]
    chunk_count = (frame_count + MODEL_FRAMES - 1) // MODEL_FRAMES
    weighted_logits = np.zeros(24, dtype=np.float64)
    for chunk_index in range(chunk_count):
        start = chunk_index * frame_count // chunk_count
        end = (chunk_index + 1) * frame_count // chunk_count
        source = spectrogram[:, start:end]
        source_frames = source.shape[1]
        indices = np.arange(MODEL_FRAMES) % source_frames
        chunk = source[:, indices]
        logits = model(torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0))[0]
        weighted_logits += logits.numpy().astype(np.float64) * source_frames
    return (weighted_logits / frame_count).astype(np.float32)


def main() -> int:
    arguments = parse_arguments()
    samples, _ = librosa.load(arguments.audio, sr=44_100, mono=True)
    spectrogram = np.log1p(
        np.abs(
            librosa.cqt(
                samples,
                sr=44_100,
                hop_length=8_820,
                n_bins=105,
                bins_per_octave=24,
                fmin=65,
            )
        )
    ).astype(np.float32)
    model = load_model(arguments.checkpoint)
    with torch.inference_mode():
        full_logits = model(torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0))[0]
        full_probabilities = torch.softmax(full_logits, dim=0).numpy()
        chunks = chunked_logits(model, spectrogram)
        chunked_probabilities = torch.softmax(torch.from_numpy(chunks), dim=0).numpy()
    print(
        json.dumps(
            {
                "frames": spectrogram.shape[1],
                "full": camelot_class(full_probabilities),
                "chunked": camelot_class(chunked_probabilities),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
