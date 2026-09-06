#!/usr/bin/env python3
"""Aggregate one evaluated instance-prediction row per case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("*/evaluation/metrics.csv"))
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            path_rows = list(csv.DictReader(handle))
        if len(path_rows) != 1:
            raise ValueError(f"Expected one row in {path}, found {len(path_rows)}")
        rows.extend(path_rows)
    cases = [row["case"] for row in rows]
    if len(rows) != args.expected_cases or len(set(cases)) != args.expected_cases:
        raise ValueError(
            f"Expected {args.expected_cases} distinct cases, found {len(rows)} rows/{len(set(cases))} cases"
        )
    methods = {row["method"] for row in rows}
    if len(methods) != 1:
        raise ValueError(f"Expected one method, found {sorted(methods)}")

    summary: dict[str, object] = {
        "method": next(iter(methods)),
        "cases": len(rows),
        "case_names": sorted(cases),
    }
    for key in ("precision", "recall", "f1", "accuracy", "panoptic_quality"):
        summary[f"macro_{key}"] = sum(float(row[key]) for row in rows) / len(rows)
    for key in ("tp", "fp", "fn"):
        summary[key] = sum(int(row[key]) for row in rows)
    tp, fp, fn = int(summary["tp"]), int(summary["fp"]), int(summary["fn"])
    summary["pooled_precision"] = tp / (tp + fp) if tp + fp else 0.0
    summary["pooled_recall"] = tp / (tp + fn) if tp + fn else 0.0
    summary["pooled_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    summary["pooled_accuracy"] = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["case"]))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
