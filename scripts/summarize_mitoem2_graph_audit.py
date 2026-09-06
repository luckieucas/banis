#!/usr/bin/env python3
"""Aggregate per-case SDT affinity graph audit CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def pooled(rows: list[dict[str, str]]) -> dict[str, float | int]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    accuracy = tp / (tp + fp + fn) if tp else 0.0
    return {
        "cases": len(rows),
        "mean_f1": sum(float(row["f1"]) for row in rows) / len(rows),
        "pooled_f1": f1,
        "pooled_accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases: dict[str, list[dict[str, str]]] = {}
    for path in sorted(args.input_root.rglob("metrics.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            cases[rows[0]["case"]] = rows
    if not cases:
        raise SystemExit(f"No metrics.csv files found below {args.input_root}")

    per_case = []
    for case, rows in sorted(cases.items()):
        baseline = next(row for row in rows if row["method"] == "baseline")
        oracle = next(row for row in rows if row["method"] == "strict_local_oracle")
        affinity_rows = [row for row in rows if row["method"] == "affinity_merge"]
        best = max(affinity_rows, key=lambda row: (float(row["f1"]), float(row["accuracy"])))
        per_case.append(
            {
                "case": case,
                "baseline_f1": float(baseline["f1"]),
                "oracle_f1": float(oracle["f1"]),
                "oracle_gain": float(oracle["f1"]) - float(baseline["f1"]),
                "best_affinity_f1_analysis_only": float(best["f1"]),
                "best_affinity_gain_analysis_only": float(best["f1"]) - float(baseline["f1"]),
                "best_threshold_analysis_only": float(best["threshold"]),
            }
        )

    by_threshold: dict[float, list[dict[str, str]]] = defaultdict(list)
    baseline_rows = []
    oracle_rows = []
    for rows in cases.values():
        for row in rows:
            if row["method"] == "baseline":
                baseline_rows.append(row)
            elif row["method"] == "strict_local_oracle":
                oracle_rows.append(row)
            elif row["method"] == "affinity_merge":
                by_threshold[float(row["threshold"])].append(row)

    summary_rows = [
        {"method": "baseline", "threshold": "", **pooled(baseline_rows)},
        {"method": "strict_local_oracle", "threshold": "", **pooled(oracle_rows)},
    ]
    for threshold, rows in sorted(by_threshold.items()):
        if len(rows) == len(cases):
            summary_rows.append({"method": "affinity_merge", "threshold": threshold, **pooled(rows)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)
    with (args.output_dir / "global_thresholds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    affinity_summaries = [row for row in summary_rows if row["method"] == "affinity_merge"]
    best_mean = max(
        affinity_summaries,
        key=lambda row: (float(row["mean_f1"]), float(row["pooled_f1"])),
        default=None,
    )
    best_pooled = max(
        affinity_summaries,
        key=lambda row: (float(row["pooled_f1"]), float(row["mean_f1"])),
        default=None,
    )
    is_validation = all("_train" in case for case in cases)
    is_transfer_test = "transfer_test" in str(args.input_root)
    is_external_confirmatory = "external_confirmatory" in str(args.input_root)
    with (args.output_dir / "SUMMARY.md").open("w") as handle:
        if is_validation:
            handle.write("# MitoEM2 held-out validation SDT affinity graph audit\n\n")
            handle.write(
                "These held-out `labelsTr` cases are used for graph-rule selection. Test evaluation must use the "
                "selected shared threshold without further tuning.\n\n"
            )
        elif is_transfer_test:
            handle.write("# MitoEM2 fixed-validation-threshold transfer test\n\n")
            handle.write(
                "All affinity-merge rows use a threshold selected on held-out validation and frozen before "
                "test evaluation; no per-case test tuning is performed.\n\n"
            )
        elif is_external_confirmatory:
            handle.write("# External confirmatory SDT affinity graph evaluation\n\n")
            handle.write(
                "The affinity rule and threshold were selected on MitoEM2 validation and frozen before "
                "evaluation on the Cardiac, Kidney, and Liver volumes. No external-volume tuning was performed.\n\n"
            )
        else:
            handle.write("# MitoEM2 cached-test SDT affinity graph audit\n\n")
            handle.write(
                "These are mechanism/headroom diagnostics on `labelsTs`. Per-case and global "
                "threshold selection is analysis-only; reportable thresholds must be selected on held-out validation.\n\n"
            )
        if best_mean is not None and best_pooled is not None:
            handle.write(
                f"Best shared threshold by mean F1: {float(best_mean['threshold']):.2f} "
                f"(mean {float(best_mean['mean_f1']):.4f}, pooled {float(best_mean['pooled_f1']):.4f}).  "
                f"Best shared threshold by pooled F1: {float(best_pooled['threshold']):.2f} "
                f"(mean {float(best_pooled['mean_f1']):.4f}, pooled {float(best_pooled['pooled_f1']):.4f}).\n\n"
            )
        suffix = "" if is_validation or is_transfer_test or is_external_confirmatory else "*"
        handle.write(
            f"| case | baseline F1 | strict local oracle F1 | best affinity F1{suffix} | threshold{suffix} |\n"
        )
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in per_case:
            handle.write(
                f"| {row['case']} | {row['baseline_f1']:.4f} | {row['oracle_f1']:.4f} | "
                f"{row['best_affinity_f1_analysis_only']:.4f} | {row['best_threshold_analysis_only']:.2f} |\n"
            )
        if not is_validation and not is_transfer_test and not is_external_confirmatory:
            handle.write("\n\\* Selected on test for diagnosis only.\n")

    print(f"Summarized {len(cases)} cases into {args.output_dir}")


if __name__ == "__main__":
    main()
