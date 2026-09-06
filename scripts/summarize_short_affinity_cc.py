#!/usr/bin/env python3
"""Select or transfer one short-affinity connected-components threshold."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def aggregate(rows: list[dict[str, str]]) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--locked-selection", type=Path)
    args = parser.parse_args()

    case_rows = {}
    for path in sorted(args.input_root.rglob("metrics.csv")):
        if args.output_dir in path.parents:
            continue
        rows = list(csv.DictReader(path.open(newline="")))
        if not rows:
            continue
        case = rows[0]["case"]
        if case in case_rows:
            raise SystemExit(f"Duplicate case {case}: {path}")
        case_rows[case] = rows
    if len(case_rows) != args.expected_cases:
        raise SystemExit(f"Expected {args.expected_cases} cases, found {len(case_rows)}")

    grouped = defaultdict(list)
    all_rows = []
    for rows in case_rows.values():
        all_rows.extend(rows)
        for row in rows:
            grouped[float(row["threshold"])].append(row)
    aggregate_rows = []
    for threshold, rows in sorted(grouped.items()):
        if len(rows) == len(case_rows):
            aggregate_rows.append({"threshold": threshold, **aggregate(rows)})
    if args.locked_selection:
        locked = float(json.loads(args.locked_selection.read_text())["selected_threshold"])
        selected = next(row for row in aggregate_rows if float(row["threshold"]) == locked)
        warning = "Threshold transferred unchanged from validation."
    else:
        selected = max(
            aggregate_rows,
            key=lambda row: (float(row["pooled_f1"]), float(row["mean_f1"]), -float(row["threshold"])),
        )
        warning = "Threshold selected by pooled validation F1, then macro F1, then lower threshold."

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_case_metrics.csv", all_rows), ("operating_points.csv", aggregate_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "input_root": str(args.input_root),
        "case_names": sorted(case_rows),
        "locked_selection": str(args.locked_selection) if args.locked_selection else "",
        "selected_threshold": float(selected["threshold"]),
        "selected_result": selected,
        "warning": warning,
    }
    (args.output_dir / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
