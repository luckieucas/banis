#!/usr/bin/env python3
"""Paired, domain-clustered uncertainty analysis for graph-repair results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_COMPARISONS = (
    "baseline:all_offsets",
    "baseline:long_only",
    "all_offsets:long_only",
    "all_offsets:short_only",
    "all_offsets:all_offsets_local_candidates",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case-metrics", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument(
        "--additional-method",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Add a transfer-result CSV containing one non-baseline row per case.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def domain_from_case(case: str) -> str:
    return case.split("_", 1)[0]


def pooled_f1(rows: list[dict[str, str]], indices: list[int]) -> float:
    tp = sum(int(rows[index]["tp"]) for index in indices)
    fp = sum(int(rows[index]["fp"]) for index in indices)
    fn = sum(int(rows[index]["fn"]) for index in indices)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def exact_cluster_sign_flip_pvalue(cluster_deltas: np.ndarray) -> float:
    nonzero = cluster_deltas[np.abs(cluster_deltas) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    if len(nonzero) > 20:
        raise ValueError("Exact sign-flip test is limited to 20 non-zero clusters")
    observed = abs(float(nonzero.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistics.append(abs(float(np.mean(nonzero * np.asarray(signs)))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return float(lower), float(upper)


def holm_adjust(rows: list[dict]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["cluster_sign_flip_p"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min((total - rank) * float(rows[index]["cluster_sign_flip_p"]), 1.0)
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def main() -> None:
    args = parse_args()
    comparisons = args.comparison or list(DEFAULT_COMPARISONS)
    selection = json.loads(args.selection.read_text())["best_by_variant"]
    with args.per_case_metrics.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    selected: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in raw_rows:
        case = row["case"]
        method = row["variant"]
        if method == "baseline":
            selected[method][case] = row
            continue
        if method not in selection:
            continue
        locked = selection[method]
        if (
            float(row["threshold"]) == float(locked["threshold"])
            and int(row["minimum_evidence"]) == int(locked["minimum_evidence"])
            and float(row["minimum_high_fraction"]) == float(locked["minimum_high_fraction"])
        ):
            selected[method][case] = row

    additional_sources = {}
    for specification in args.additional_method:
        if "=" not in specification:
            raise SystemExit(f"Invalid --additional-method {specification!r}; expected NAME=CSV")
        name, path_text = specification.split("=", 1)
        path = Path(path_text)
        additional_sources[name] = str(path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            method = row.get("method", row.get("variant", ""))
            if method == "baseline":
                continue
            case = row["case"]
            if case in selected[name]:
                raise SystemExit(f"Multiple non-baseline rows for {name}:{case} in {path}")
            selected[name][case] = row

    rng = np.random.default_rng(args.seed)
    result_rows = []
    for comparison in comparisons:
        reference_name, candidate_name = comparison.split(":", 1)
        reference_by_case = selected[reference_name]
        candidate_by_case = selected[candidate_name]
        cases = sorted(set(reference_by_case) & set(candidate_by_case))
        if len(cases) != len(reference_by_case) or len(cases) != len(candidate_by_case):
            raise SystemExit(f"Case mismatch for {comparison}")
        reference_rows = [reference_by_case[case] for case in cases]
        candidate_rows = [candidate_by_case[case] for case in cases]
        differences = np.asarray(
            [float(candidate["f1"]) - float(reference["f1"]) for reference, candidate in zip(reference_rows, candidate_rows)]
        )

        cluster_indices: dict[str, list[int]] = defaultdict(list)
        for index, case in enumerate(cases):
            cluster_indices[domain_from_case(case)].append(index)
        clusters = sorted(cluster_indices)
        cluster_deltas = np.asarray([differences[cluster_indices[cluster]].mean() for cluster in clusters])
        bootstrap_macro = np.empty(args.bootstrap_replicates, dtype=np.float64)
        bootstrap_pooled = np.empty(args.bootstrap_replicates, dtype=np.float64)
        for replicate in range(args.bootstrap_replicates):
            sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
            indices = [index for cluster in sampled_clusters for index in cluster_indices[cluster]]
            bootstrap_macro[replicate] = float(differences[indices].mean())
            bootstrap_pooled[replicate] = pooled_f1(candidate_rows, indices) - pooled_f1(reference_rows, indices)

        macro_ci = percentile_interval(bootstrap_macro)
        pooled_ci = percentile_interval(bootstrap_pooled)
        result_rows.append(
            {
                "reference": reference_name,
                "candidate": candidate_name,
                "cases": len(cases),
                "domains": len(clusters),
                "macro_f1_delta": float(differences.mean()),
                "macro_f1_ci95_low": macro_ci[0],
                "macro_f1_ci95_high": macro_ci[1],
                "pooled_f1_delta": pooled_f1(candidate_rows, list(range(len(cases))))
                - pooled_f1(reference_rows, list(range(len(cases)))),
                "pooled_f1_ci95_low": pooled_ci[0],
                "pooled_f1_ci95_high": pooled_ci[1],
                "wins": int(np.count_nonzero(differences > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                "losses": int(np.count_nonzero(differences < -1e-12)),
                "cluster_sign_flip_p": exact_cluster_sign_flip_pvalue(cluster_deltas),
                "holm_adjusted_p": 0.0,
            }
        )

    holm_adjust(result_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    metadata = {
        "per_case_metrics": str(args.per_case_metrics),
        "selection": str(args.selection),
        "additional_sources": additional_sources,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "domain cluster; all volumes from a sampled domain are retained",
        "randomization_test": "exact two-sided sign flip of domain-mean paired F1 deltas",
        "multiplicity": "Holm correction across declared comparisons",
        "seed": args.seed,
        "results": result_rows,
    }
    (args.output_dir / "paired_statistics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
