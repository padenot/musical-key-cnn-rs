from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx


LOGGER = logging.getLogger("musical_key_cnn.debug_model")
PROBED_OPERATORS = {"Conv", "Elu", "MaxPool", "GlobalAveragePool"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose MusicalKeyCNN intermediate tensors for backend parity checks."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    model = onnx.shape_inference.infer_shapes(onnx.load(arguments.input))
    value_info = {
        value.name: value
        for value in (*model.graph.value_info, *model.graph.output)
    }
    selected_names = [
        output
        for node in model.graph.node
        if node.op_type in PROBED_OPERATORS
        for output in node.output
    ]
    for name in selected_names:
        descriptor = value_info.get(name)
        if descriptor is None:
            raise RuntimeError(f"shape inference returned no descriptor for {name}")
        model.graph.output.append(descriptor)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, arguments.output)
    LOGGER.info("exposed %d intermediates in %s", len(selected_names), arguments.output)
    for name in selected_names:
        LOGGER.info("%s", name)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("debug model export failed")
        raise SystemExit(1)
