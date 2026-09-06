#!/usr/bin/env python3
"""Evaluate cached BANIS short/long affinity and candidate-graph ablations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_graph import (  # noqa: E402
    affinity_merge_lut,
    build_fragment_contingency,
    load_volume,
    metrics_from_contingency,
    pool_affinity_evidence_channels,
)


VARIANTS = {
    "short_only": {"channels": (0, 1, 2), "require_short_range_support": False},
    "long_only": {"channels": (3, 4, 5), "require_short_range_support": False},
    "all_offsets": {"channels": (0, 1, 2, 3, 4, 5), "require_short_range_support": False},
    "all_offsets_local_candidates": {
        "channels": (0, 1, 2, 3, 4, 5),
        "require_short_range_support": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    parser.add_argument("--minimum-evidence", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--minimum-high-fraction", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--block-shape", type=int, nargs=3, default=(32, 256, 256))
    parser.add_argument(
        "--fixed-setting",
        nargs=3,
        action="append",
        metavar=("THRESHOLD", "MINIMUM_EVIDENCE", "MINIMUM_HIGH_FRACTION"),
        help="Evaluate only explicit settings; may be repeated. Overrides the Cartesian sweep.",
    )
    parser.add_argument(
        "--settings-from",
        type=Path,
        help="Validation selection JSON whose per-variant settings are evaluated without test tuning.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
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


def requested_settings(args: argparse.Namespace) -> list[tuple[float, int, float]]:
    if args.settings_from:
        if args.fixed_setting:
            raise ValueError("--settings-from and --fixed-setting are mutually exclusive")
        selection = json.loads(args.settings_from.read_text())
        selected_rows = selection["best_by_variant"]
        settings = {
            (
                float(row["threshold"]),
                int(row["minimum_evidence"]),
                float(row["minimum_high_fraction"]),
            )
            for row in selected_rows.values()
        }
        return sorted(settings)
    if args.fixed_setting:
        return [
            (float(threshold), int(minimum_evidence), float(minimum_high_fraction))
            for threshold, minimum_evidence, minimum_high_fraction in args.fixed_setting
        ]
    return [
        (threshold, minimum_evidence, minimum_high_fraction)
        for threshold in args.thresholds
        for minimum_evidence in args.minimum_evidence
        for minimum_high_fraction in args.minimum_high_fraction
    ]


def main() -> None:
    args = parse_args()
    settings = requested_settings(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict] = []
    edge_counts: list[dict] = []

    for graph_dir in args.graph_dir:
        config = json.loads((graph_dir / "config.json").read_text())
        with (graph_dir / "edge_evidence.csv").open(newline="") as handle:
            cached_edges = list(csv.DictReader(handle))
        fragments = load_volume(Path(config["fragments"]))
        ground_truth = load_volume(Path(config["ground_truth"]))
        mask_path = config.get("mask")
        mask = load_volume(Path(mask_path)) if mask_path else None
        contingency = build_fragment_contingency(
            ground_truth,
            fragments,
            mask=mask,
            block_shape=args.block_shape,
        )
        n_labels = max(contingency.fragment_areas, default=0) + 1
        baseline = metrics_from_contingency(contingency)
        case_rows.append(
            {
                "case": config["case"],
                "variant": "baseline",
                "threshold": "",
                "minimum_evidence": "",
                "minimum_high_fraction": "",
                "accepted_merges": 0,
                **baseline,
            }
        )

        for variant, specification in VARIANTS.items():
            edges = pool_affinity_evidence_channels(cached_edges, **specification)
            edge_counts.append(
                {
                    "case": config["case"],
                    "variant": variant,
                    "candidate_edges": len(edges),
                }
            )
            for threshold, minimum_evidence, minimum_high_fraction in settings:
                lut, accepted = affinity_merge_lut(
                    n_labels,
                    edges,
                    threshold=threshold,
                    minimum_evidence=minimum_evidence,
                    minimum_high_fraction=minimum_high_fraction,
                )
                metrics = metrics_from_contingency(contingency, label_lut=lut)
                case_rows.append(
                    {
                        "case": config["case"],
                        "variant": variant,
                        "threshold": threshold,
                        "minimum_evidence": minimum_evidence,
                        "minimum_high_fraction": minimum_high_fraction,
                        "accepted_merges": accepted,
                        **metrics,
                    }
                )
        print(f"Evaluated {config['case']}: {len(cached_edges)} cached edges", flush=True)

    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in case_rows:
        grouped[
            (
                str(row["variant"]),
                str(row["threshold"]),
                str(row["minimum_evidence"]),
                str(row["minimum_high_fraction"]),
            )
        ].append(row)
    summary_rows = []
    for (variant, threshold, minimum_evidence, minimum_high_fraction), rows in grouped.items():
        if len(rows) != len(args.graph_dir):
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
    best_by_variant = {}
    for variant in VARIANTS:
        candidates = [row for row in summary_rows if row["variant"] == variant]
        best_by_variant[variant] = max(
            candidates,
            key=lambda row: (float(row["pooled_f1"]), float(row["mean_f1"])),
        )

    write_csv(args.output_dir / "per_case_metrics.csv", case_rows)
    write_csv(args.output_dir / "candidate_edge_counts.csv", edge_counts)
    write_csv(args.output_dir / "operating_points.csv", summary_rows)
    metadata = {
        "graph_dirs": [str(path) for path in args.graph_dir],
        "variants": VARIANTS,
        "settings": [
            {
                "threshold": threshold,
                "minimum_evidence": minimum_evidence,
                "minimum_high_fraction": minimum_high_fraction,
            }
            for threshold, minimum_evidence, minimum_high_fraction in settings
        ],
        "settings_from": str(args.settings_from) if args.settings_from else "",
        "best_by_variant_pooled_f1": best_by_variant,
        "warning": "Hyperparameters selected here may only be transferred to a test set when these inputs are the declared validation set.",
    }
    (args.output_dir / "ablation_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(best_by_variant, indent=2))


if __name__ == "__main__":
    main()
