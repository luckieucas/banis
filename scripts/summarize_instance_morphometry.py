#!/usr/bin/env python3
"""Summarize locked per-case morphometry and its preregistered primary endpoint."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ("sdt_fragments", "region_mws", "banis_gr", "banis_gr_plus")
METRICS = (
    "count_absolute_relative_error",
    "count_signed_relative_error",
    "mean_gt_best_overlap_fraction",
    "mean_pred_best_purity",
    "volume_distribution_wasserstein_log",
    "elongation_distribution_wasserstein_log",
    "matched_mean_absolute_volume_log_error",
    "matched_mean_absolute_elongation_log_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def domain_from_case(case: str) -> str:
    return case.split("_", 1)[0]


def read_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(root.glob("*/method_metrics.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1.0e-12]
    if not len(nonzero):
        return 1.0
    observed = abs(float(np.mean(nonzero)))
    randomized = [
        abs(float(np.mean(nonzero * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero))
    ]
    return float(np.mean(np.asarray(randomized) >= observed - 1.0e-15))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_root)
    cases = sorted({row["case"] for row in rows})
    if len(cases) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} cases, found {len(cases)}: {cases}")
    by_method_case = {(row["method"], row["case"]): row for row in rows}
    expected = {(method, case) for method in METHODS for case in cases}
    if set(by_method_case) != expected or len(rows) != len(expected):
        missing = sorted(expected - set(by_method_case))
        extra = sorted(set(by_method_case) - expected)
        raise RuntimeError(f"Incomplete or duplicate method/case rows; missing={missing}, extra={extra}")

    summary_rows = []
    for method in METHODS:
        selected = [by_method_case[(method, case)] for case in cases]
        summary_rows.append(
            {
                "method": method,
                "cases": len(selected),
                "macro_instance_f1_check": float(
                    np.mean([float(row["instance_f1_check"]) for row in selected])
                ),
                **{
                    f"macro_{metric}": float(np.mean([float(row[metric]) for row in selected]))
                    for metric in METRICS
                },
            }
        )

    # Single preregistered inferential endpoint: reduction in per-volume absolute
    # relative object-count error for BANIS-GR+ versus SDT fragments.
    reference = np.asarray(
        [float(by_method_case[("sdt_fragments", case)]["count_absolute_relative_error"]) for case in cases]
    )
    candidate = np.asarray(
        [float(by_method_case[("banis_gr_plus", case)]["count_absolute_relative_error"]) for case in cases]
    )
    benefit = reference - candidate
    cluster_indices: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        cluster_indices[domain_from_case(case)].append(index)
    clusters = sorted(cluster_indices)
    cluster_benefits = np.asarray(
        [float(np.mean(benefit[cluster_indices[cluster]])) for cluster in clusters]
    )
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap_replicates, dtype=np.float64)
    for replicate in range(args.bootstrap_replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = [index for cluster in sampled for index in cluster_indices[cluster]]
        bootstrap[replicate] = float(np.mean(benefit[indices]))
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    primary = {
        "reference": "sdt_fragments",
        "candidate": "banis_gr_plus",
        "metric": "count_absolute_relative_error",
        "direction": "positive benefit means lower error for candidate",
        "cases": len(cases),
        "domains": len(clusters),
        "reference_macro": float(np.mean(reference)),
        "candidate_macro": float(np.mean(candidate)),
        "mean_benefit": float(np.mean(benefit)),
        "domain_cluster_bootstrap_ci95_low": float(low),
        "domain_cluster_bootstrap_ci95_high": float(high),
        "wins": int(np.count_nonzero(benefit > 1.0e-12)),
        "ties": int(np.count_nonzero(np.abs(benefit) <= 1.0e-12)),
        "losses": int(np.count_nonzero(benefit < -1.0e-12)),
        "exact_domain_sign_flip_p": exact_sign_flip_pvalue(cluster_benefits),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "method_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "per_case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["case"], row["method"])))
    record = {
        "cohort": args.cohort,
        "analysis_role": "locked downstream analysis",
        "case_names": cases,
        "methods": METHODS,
        "primary_endpoint": primary,
        "secondary_endpoints": [metric for metric in METRICS if metric != "count_absolute_relative_error"],
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "domain cluster; all cases from a sampled domain retained",
        "randomization_test": "exact two-sided sign flip of domain-mean paired benefits",
        "seed": args.seed,
        "summaries": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
