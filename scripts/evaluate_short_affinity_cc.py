#!/usr/bin/env python3
"""Evaluate a conventional short-affinity connected-components baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_graph import instance_metrics, load_volume  # noqa: E402
from src.inference.short_affinity_cc import short_affinity_components  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--threshold-from", type=Path)
    parser.add_argument("--minimum-size", type=int, default=200)
    parser.add_argument("--block-shape", type=int, nargs=3, default=(32, 256, 256))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.threshold_from:
        selection = json.loads(args.threshold_from.read_text())
        args.thresholds = [float(selection["selected_threshold"])]
    config = json.loads((args.graph_dir / "config.json").read_text())
    affinity_paths = [Path(config["affinities"][str(channel)]) for channel in range(3)]
    arrays = [zarr.open_array(str(path), mode="r") for path in affinity_paths]
    shape = tuple(arrays[0].shape)
    if any(tuple(array.shape) != shape for array in arrays):
        raise ValueError("Short-affinity channel shapes do not match")
    probabilities = np.empty((3, *shape), dtype=np.float16)
    for channel, array in enumerate(arrays):
        probabilities[channel] = array[:]
        print(f"Loaded channel {channel}: {affinity_paths[channel]}", flush=True)

    ground_truth = load_volume(config["ground_truth"])
    mask = load_volume(config["mask"]) if config.get("mask") else None
    rows = []
    for threshold in args.thresholds:
        threshold_started = time.time()
        segmentation = short_affinity_components(probabilities, threshold, args.minimum_size)
        metrics = instance_metrics(
            ground_truth,
            segmentation,
            mask=mask,
            block_shape=args.block_shape,
        )
        rows.append(
            {
                "case": config["case"],
                "method": "short_affinity_cc",
                "threshold": threshold,
                "minimum_size": args.minimum_size,
                "elapsed_seconds": time.time() - threshold_started,
                **metrics,
            }
        )
        print(f"threshold={threshold:.2f} f1={metrics['f1']:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "case": config["case"],
        "graph_dir": str(args.graph_dir),
        "affinity_paths": [str(path) for path in affinity_paths],
        "thresholds": args.thresholds,
        "threshold_from": str(args.threshold_from) if args.threshold_from else "",
        "minimum_size": args.minimum_size,
        "block_shape": args.block_shape,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")


if __name__ == "__main__":
    main()
