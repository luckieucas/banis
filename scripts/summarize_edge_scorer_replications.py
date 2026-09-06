#!/usr/bin/env python3
"""Summarize frozen edge-scorer bootstrap replications across seeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1")
OUTPUT = ROOT / "learned_edge_scorer_stage1" / "replication_summary"
METRICS = ("mean_f1", "mean_panoptic_quality", "pooled_f1", "pooled_accuracy")


def summary_path(variant: str, split: str, seed: int) -> Path:
    if split == "test":
        directory = "test_transfer" if variant == "anchored" else "test_transfer_unanchored"
        return ROOT / "learned_edge_scorer_stage1" / directory / f"seed{seed}" / "evaluation_summary.json"
    base = "learned_gate_summary" if variant == "anchored" else "learned_unanchored_summary"
    directory = base if seed == 0 else f"{base}_seed{seed}"
    return ROOT / "external_confirmatory" / directory / "evaluation_summary.json"


def main() -> None:
    seed_rows = []
    aggregate_rows = []
    for variant in ("anchored", "unanchored"):
        for split in ("test", "external"):
            group = []
            for seed in range(3):
                path = summary_path(variant, split, seed)
                metadata = json.loads(path.read_text())
                result = metadata["best_operating_point"]
                row = {
                    "variant": variant,
                    "split": split,
                    "seed": seed,
                    "probability_threshold": float(result["probability_threshold"]),
                    "maximum_uncertainty": float(result["maximum_uncertainty"]),
                    **{metric: float(result[metric]) for metric in METRICS},
                    "source": str(path),
                }
                seed_rows.append(row)
                group.append(row)
            aggregate = {"variant": variant, "split": split, "seeds": len(group)}
            for metric in METRICS:
                values = np.asarray([row[metric] for row in group], dtype=np.float64)
                aggregate[f"{metric}_mean"] = float(values.mean())
                aggregate[f"{metric}_sample_sd"] = float(values.std(ddof=1))
            aggregate_rows.append(aggregate)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with (OUTPUT / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    metadata = {
        "replication_unit": "volume-bootstrap logistic edge-scorer ensemble seed; dense predictor fixed",
        "ddof": 1,
        "per_seed": seed_rows,
        "aggregate": aggregate_rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
