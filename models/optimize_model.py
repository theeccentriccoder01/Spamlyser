#!/usr/bin/env python3
"""
Model optimisation and export script with configurable paths and formats.

Usage examples::

    # Export to ONNX with default settings
    python -m models.optimize_model

    # Export to TorchScript
    python -m models.optimize_model --model-id bert-base-uncased \\
        --output ./models/exports/bert.pt --format torchscript

    # Custom sequence length
    python -m models.optimize_model --seq-len 128 --opset 17
"""

import argparse
import sys
from pathlib import Path

import torch


def _get_model(model_id: str):
    from transformers import AutoModelForSequenceClassification

    print(f"Loading {model_id} ...")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return model


def export_onnx(model, output: Path, seq_len: int, opset: int):
    dummy = torch.randint(0, 100, (1, seq_len))
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["input_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
            },
            opset_version=opset,
        )
    print(f"ONNX model saved to {output}")


def export_torchscript(model, output: Path, seq_len: int):
    dummy = torch.randint(0, 100, (1, seq_len))
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
        traced.save(str(output))
    print(f"TorchScript model saved to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Export or optimise a HuggingFace model"
    )
    parser.add_argument(
        "--model-id",
        default="distilbert-base-uncased",
        help="HuggingFace model identifier (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("distilbert.onnx"),
        help="Output file path (default: distilbert.onnx)",
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "torchscript"],
        default="onnx",
        help="Export format (default: onnx)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Dummy input sequence length (default: 8)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    args = parser.parse_args()

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    model = _get_model(args.model_id)

    if args.format == "onnx":
        export_onnx(model, output, args.seq_len, args.opset)
    elif args.format == "torchscript":
        export_torchscript(model, output, args.seq_len)


if __name__ == "__main__":
    main()
