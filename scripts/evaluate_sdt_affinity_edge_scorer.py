#!/usr/bin/env python3
"""Evaluate a learned edge scorer on explicitly supplied graph-audit cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_edge_scorer import (  # noqa: E402
    EDGE_FEATURE_NAMES,
    edge_feature_matrix,
    scored_merge_lut,
)
from src.inference.sdt_affinity_graph import (  # noqa: E402
    build_fragment_contingency,
    load_volume,
    metrics_from_contingency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--maximum-uncertainties", type=float, nargs="+", default=[0.05, 0.1, 0.2, 1.0])
    parser.add_argument("--minimum-evidence", type=int, default=8)
    parser.add_argument("--minimum-affinity-mean", type=float, default=0.0)
    parser.add_argument("--minimum-high-fraction", type=float, default=0.0)
    parser.add_argument("--block-shape", type=int, nargs=3, default=(32, 256, 256))
    parser.add_argument(
        "--operating-point-from",
        type=Path,
        help="Validation evaluation_summary.json; evaluate only its frozen best operating point.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("pooled_f1", "mean_f1"),
        default="pooled_f1",
        help="Used only to identify the best row in this explicitly supplied evaluation set.",
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
    pooled_f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    pooled_accuracy = tp / (tp + fp + fn) if tp else 0.0
    return {
        "cases": len(rows),
        "mean_f1": sum(float(row["f1"]) for row in rows) / len(rows),
        "mean_panoptic_quality": sum(float(row["panoptic_quality"]) for row in rows) / len(rows),
        "pooled_f1": pooled_f1,
        "pooled_accuracy": pooled_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    if args.operating_point_from:
        locked = json.loads(args.operating_point_from.read_text())["best_operating_point"]
        args.probability_thresholds = [float(locked["probability_threshold"])]
        args.maximum_uncertainties = [float(locked["maximum_uncertainty"])]
    scorer = joblib.load(args.model)
    if tuple(scorer.feature_names) != EDGE_FEATURE_NAMES:
        raise SystemExit("Model feature schema does not match the current implementation")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_rows = []
    scored_edge_rows = []
    for graph_dir in args.graph_dir:
        config = json.loads((graph_dir / "config.json").read_text())
        with (graph_dir / "edge_evidence.csv").open(newline="") as handle:
            edges = list(csv.DictReader(handle))
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
        features = edge_feature_matrix(
            edges,
            contingency.fragment_areas,
            config["voxel_spacing_nm_zyx"],
        )
        probability, uncertainty = scorer.predict_distribution(features)
        for row, score, spread in zip(edges, probability, uncertainty):
            scored_edge_rows.append(
                {
                    "case": config["case"],
                    "fragment_a": row["fragment_a"],
                    "fragment_b": row["fragment_b"],
                    "probability": float(score),
                    "uncertainty": float(spread),
                    "evidence_count": row["count"],
                }
            )
        baseline = metrics_from_contingency(contingency)
        case_rows.append(
            {
                "case": config["case"],
                "method": "baseline",
                "probability_threshold": "",
                "maximum_uncertainty": "",
                "accepted_merges": 0,
                **baseline,
            }
        )
        for probability_threshold in args.probability_thresholds:
            for maximum_uncertainty in args.maximum_uncertainties:
                lut, accepted = scored_merge_lut(
                    n_labels,
                    edges,
                    probability,
                    uncertainty,
                    probability_threshold=probability_threshold,
                    maximum_uncertainty=maximum_uncertainty,
                    minimum_evidence=args.minimum_evidence,
                    minimum_affinity_mean=args.minimum_affinity_mean,
                    minimum_high_fraction=args.minimum_high_fraction,
                )
                metrics = metrics_from_contingency(contingency, label_lut=lut)
                case_rows.append(
                    {
                        "case": config["case"],
                        "method": "learned_edge_scorer",
                        "probability_threshold": probability_threshold,
                        "maximum_uncertainty": maximum_uncertainty,
                        "accepted_merges": accepted,
                        **metrics,
                    }
                )
        print(f"Evaluated {config['case']}: {len(edges)} candidate edges")

    by_setting: dict[tuple[str, str], list[dict]] = defaultdict(list)
    baseline_rows = []
    for row in case_rows:
        if row["method"] == "baseline":
            baseline_rows.append(row)
        else:
            by_setting[(str(row["probability_threshold"]), str(row["maximum_uncertainty"]))].append(row)
    summary_rows = [{"method": "baseline", "probability_threshold": "", "maximum_uncertainty": "", **aggregate(baseline_rows)}]
    for (probability_threshold, maximum_uncertainty), rows in sorted(
        by_setting.items(), key=lambda item: (float(item[0][0]), float(item[0][1]))
    ):
        if len(rows) == len(args.graph_dir):
            summary_rows.append(
                {
                    "method": "learned_edge_scorer",
                    "probability_threshold": probability_threshold,
                    "maximum_uncertainty": maximum_uncertainty,
                    **aggregate(rows),
                }
            )
    candidates = [row for row in summary_rows if row["method"] == "learned_edge_scorer"]
    best = max(
        candidates,
        key=lambda row: (
            float(row[args.selection_objective]),
            float(row["mean_f1"] if args.selection_objective == "pooled_f1" else row["pooled_f1"]),
            -float(row["probability_threshold"]),
            float(row["maximum_uncertainty"]),
        ),
    )
    write_csv(args.output_dir / "per_case_metrics.csv", case_rows)
    write_csv(args.output_dir / "edge_scores.csv", scored_edge_rows)
    write_csv(args.output_dir / "operating_points.csv", summary_rows)
    metadata = {
        "model": str(args.model),
        "graph_dirs": [str(path) for path in args.graph_dir],
        "probability_thresholds": args.probability_thresholds,
        "maximum_uncertainties": args.maximum_uncertainties,
        "minimum_evidence": args.minimum_evidence,
        "minimum_affinity_mean": args.minimum_affinity_mean,
        "minimum_high_fraction": args.minimum_high_fraction,
        "block_shape": args.block_shape,
        "selection_objective": f"{args.selection_objective}_then_alternate_f1_then_minimum_gating",
        "operating_point_from": str(args.operating_point_from) if args.operating_point_from else "",
        "best_operating_point": best,
        "warning": (
            "Operating point transferred unchanged from the supplied validation summary."
            if args.operating_point_from
            else "An operating point is reportable only when these graph directories are a declared validation set."
        ),
    }
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["best_operating_point"], indent=2))


if __name__ == "__main__":
    main()
