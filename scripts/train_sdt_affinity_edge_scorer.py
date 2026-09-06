#!/usr/bin/env python3
"""Train a volume-bootstrap probabilistic scorer from exported edge tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_edge_scorer import (  # noqa: E402
    EDGE_FEATURE_NAMES,
    BootstrapLogisticEdgeScorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-table", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=25)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for path in args.edge_table:
        with path.open(newline="") as handle:
            table = list(csv.DictReader(handle))
        records.extend(row for row in table if int(row["label_valid"]) == 1)
    if not records:
        raise SystemExit("No valid labeled edges found")

    features = np.asarray(
        [[float(row[f"feature_{name}"]) for name in EDGE_FEATURE_NAMES] for row in records],
        dtype=np.float64,
    )
    targets = np.asarray([int(row["target_merge"]) for row in records], dtype=np.int8)
    groups = np.asarray([row["case"] for row in records])
    scorer = BootstrapLogisticEdgeScorer(
        n_estimators=args.n_estimators,
        regularization_c=args.regularization_c,
        random_state=args.seed,
    ).fit(features, targets, groups=groups)
    probability, uncertainty = scorer.predict_distribution(features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "edge_scorer.joblib"
    joblib.dump(scorer, model_path)
    metrics = {
        "training_edges": len(records),
        "positive_edges": int(targets.sum()),
        "negative_edges": int((1 - targets).sum()),
        "training_cases": sorted(set(groups)),
        "n_estimators": args.n_estimators,
        "regularization_c": args.regularization_c,
        "seed": args.seed,
        "feature_names": list(EDGE_FEATURE_NAMES),
        "sklearn_version": sklearn.__version__,
        "apparent_roc_auc": float(roc_auc_score(targets, probability)),
        "apparent_average_precision": float(average_precision_score(targets, probability)),
        "apparent_brier": float(brier_score_loss(targets, probability)),
        "mean_uncertainty": float(uncertainty.mean()),
        "note": "Apparent training metrics are implementation diagnostics, not reportable validation results.",
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved scorer to {model_path}")


if __name__ == "__main__":
    main()
