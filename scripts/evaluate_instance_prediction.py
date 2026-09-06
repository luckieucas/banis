#!/usr/bin/env python3
"""Evaluate an existing 3-D instance prediction with the BANIS TMI metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_graph import instance_metrics, load_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-shape", type=int, nargs=3, default=(32, 256, 256))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    prediction = load_volume(args.prediction)
    ground_truth = load_volume(args.ground_truth)
    mask = load_volume(args.mask) if args.mask else None
    if tuple(prediction.shape) != tuple(ground_truth.shape):
        raise ValueError(
            f"Prediction/ground-truth shape mismatch: {prediction.shape} != {ground_truth.shape}"
        )
    prediction_array = np.asarray(prediction)
    if prediction_array.ndim != 3 or not np.issubdtype(prediction_array.dtype, np.integer):
        raise ValueError(f"Expected a 3-D integer instance prediction, got {prediction_array.dtype}")
    if prediction_array.size and int(prediction_array.min()) < 0:
        raise ValueError("Instance labels must be non-negative")
    metrics = instance_metrics(
        ground_truth,
        prediction_array,
        mask=mask,
        block_shape=args.block_shape,
    )
    row = {
        "case": args.case,
        "method": args.method,
        "elapsed_seconds": time.time() - started,
        **metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    metadata = {
        "case": args.case,
        "method": args.method,
        "prediction": str(args.prediction),
        "ground_truth": str(args.ground_truth),
        "mask": str(args.mask) if args.mask else "",
        "shape": list(prediction_array.shape),
        "block_shape": list(args.block_shape),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
