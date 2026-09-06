#!/usr/bin/env python3
"""Aggregate independently evaluated offset-ablation cases."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def aggregate(rows: list[dict[str, str]]) -> dict[str, float | int]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    return {
        "cases": len(rows),
        "mean_f1": sum(float(row["f1"]) for row in rows) / len(rows),
        "mean_panoptic_quality": sum(float(row["panoptic_quality"]) for row in rows) / len(rows),
        "pooled_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "pooled_accuracy": tp / (tp + fp + fn) if tp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument(
        "--locked-selection",
        type=Path,
        help="Validation selection JSON. Report exactly its settings instead of selecting on these inputs.",
    )
    args = parser.parse_args()

    by_case: dict[str, list[dict[str, str]]] = {}
    for path in sorted(args.input_root.rglob("per_case_metrics.csv")):
        if args.output_dir in path.parents:
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        case = rows[0]["case"]
        if case in by_case:
            raise SystemExit(f"Duplicate case {case}: {path}")
        by_case[case] = rows
    if not by_case:
        raise SystemExit(f"No per_case_metrics.csv found below {args.input_root}")
    if args.expected_cases is not None and len(by_case) != args.expected_cases:
        raise SystemExit(f"Expected {args.expected_cases} cases, found {len(by_case)}")

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    all_case_rows = []
    for rows in by_case.values():
        all_case_rows.extend(rows)
        for row in rows:
            grouped[
                (
                    row["variant"],
                    row["threshold"],
                    row["minimum_evidence"],
                    row["minimum_high_fraction"],
                )
            ].append(row)

    summary_rows = []
    for (variant, threshold, minimum_evidence, minimum_high_fraction), rows in grouped.items():
        if len(rows) != len(by_case):
            continue
        summary_rows.append(
            {
                "variant": variant,
                "threshold": threshold,
                "minimum_evidence": minimum_evidence,
                "minimum_high_fraction": minimum_high_fraction,
                **aggregate(rows),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            row["variant"],
            float(row["threshold"] or -1),
            int(row["minimum_evidence"] or -1),
            float(row["minimum_high_fraction"] or -1),
        )
    )
    variants = sorted({row["variant"] for row in summary_rows if row["variant"] != "baseline"})
    if args.locked_selection:
        validation_selection = json.loads(args.locked_selection.read_text())
        best_by_variant = {}
        for variant, locked in validation_selection["best_by_variant"].items():
            matching = [
                row
                for row in summary_rows
                if row["variant"] == variant
                and float(row["threshold"]) == float(locked["threshold"])
                and int(row["minimum_evidence"]) == int(locked["minimum_evidence"])
                and float(row["minimum_high_fraction"]) == float(locked["minimum_high_fraction"])
            ]
            if len(matching) != 1:
                raise SystemExit(f"Expected one locked result for {variant}, found {len(matching)}")
            best_by_variant[variant] = matching[0]
        selection_objective = "locked_validation_transfer"
    else:
        best_by_variant = {
            variant: max(
                (row for row in summary_rows if row["variant"] == variant),
                key=lambda row: (
                    float(row["pooled_f1"]),
                    float(row["mean_f1"]),
                    -abs(float(row["threshold"]) - 0.4),
                    -abs(int(row["minimum_evidence"]) - 8),
                    -abs(float(row["minimum_high_fraction"]) - 0.5),
                ),
            )
            for variant in variants
        }
        selection_objective = "pooled_f1_then_mean_f1_with_canonical_tie_break"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case_metrics.csv", all_case_rows)
    write_csv(args.output_dir / "operating_points.csv", summary_rows)
    selection = {
        "input_root": str(args.input_root),
        "cases": len(by_case),
        "case_names": sorted(by_case),
        "selection_objective": selection_objective,
        "locked_selection": str(args.locked_selection) if args.locked_selection else "",
        "best_by_variant": best_by_variant,
    }
    output_name = "locked_transfer_results.json" if args.locked_selection else "validation_selection.json"
    (args.output_dir / output_name).write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
