#!/usr/bin/env python3
"""Select region-graph MWS beta on validation and summarize all cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--locked-beta", type=float)
    return parser.parse_args()


def read_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/metrics.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], beta: float) -> dict:
    chosen = [row for row in rows if abs(float(row["beta"]) - beta) < 1e-8]
    tp = sum(int(row["tp"]) for row in chosen)
    fp = sum(int(row["fp"]) for row in chosen)
    fn = sum(int(row["fn"]) for row in chosen)
    matched_iou = sum(float(row["mean_matched_score"]) * int(row["tp"]) for row in chosen)
    pooled_precision = tp / (tp + fp) if tp else 0.0
    pooled_recall = tp / (tp + fn) if tp else 0.0
    pooled_f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    pooled_accuracy = tp / (tp + fp + fn) if tp else 0.0
    pooled_pq = matched_iou / (tp + 0.5 * fp + 0.5 * fn) if tp else 0.0
    return {
        "beta": beta,
        "cases": len(chosen),
        "macro_precision": float(np.mean([float(row["precision"]) for row in chosen])),
        "macro_recall": float(np.mean([float(row["recall"]) for row in chosen])),
        "macro_f1": float(np.mean([float(row["f1"]) for row in chosen])),
        "macro_accuracy": float(np.mean([float(row["accuracy"]) for row in chosen])),
        "macro_panoptic_quality": float(np.mean([float(row["panoptic_quality"]) for row in chosen])),
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "pooled_f1": pooled_f1,
        "pooled_accuracy": pooled_accuracy,
        "pooled_panoptic_quality": pooled_pq,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_root)
    cases = sorted({row["case"] for row in rows})
    if len(cases) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} cases, found {len(cases)}: {cases}")
    beta_values = sorted({float(row["beta"]) for row in rows})
    summaries = [aggregate(rows, beta) for beta in beta_values]
    if any(item["cases"] != args.expected_cases for item in summaries):
        raise RuntimeError("At least one beta is missing a case")
    if args.locked_beta is None:
        selected = max(summaries, key=lambda item: (item["pooled_f1"], item["macro_f1"], -item["beta"]))
        selection_role = "validation_selected"
    else:
        candidates = [item for item in summaries if abs(item["beta"] - args.locked_beta) < 1e-8]
        if len(candidates) != 1:
            raise RuntimeError(f"Locked beta {args.locked_beta} is not available exactly once")
        selected = candidates[0]
        selection_role = "locked_transfer"
    selected_rows = [row for row in rows if abs(float(row["beta"]) - selected["beta"]) < 1e-8]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "beta_summary.csv", summaries)
    write_rows(args.output_dir / "per_case_selected.csv", selected_rows)
    record = {
        "method": "region_mutex_watershed",
        "selection_role": selection_role,
        "objective": "pooled_f1; macro_f1 then lower beta break exact ties",
        "expected_cases": args.expected_cases,
        "case_names": cases,
        "selected": selected,
    }
    (args.output_dir / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
