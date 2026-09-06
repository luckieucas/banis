#!/usr/bin/env python3
"""Summarize standardized logistic coefficients across bootstrap members."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scorer = joblib.load(args.model)
    coefficients = np.vstack(
        [estimator.named_steps["logisticregression"].coef_[0] for estimator in scorer.estimators_]
    )
    rows = []
    for index, name in enumerate(scorer.feature_names):
        values = coefficients[:, index]
        rows.append(
            {
                "feature": name,
                "coefficient_mean": float(values.mean()),
                "coefficient_std": float(values.std()),
                "absolute_mean": float(np.abs(values).mean()),
                "positive_fraction": float(np.mean(values > 0)),
            }
        )
    rows.sort(key=lambda row: row["absolute_mean"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "standardized_coefficients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": str(args.model),
        "estimators": len(scorer.estimators_),
        "note": "Coefficients act on per-bootstrap standardized features; magnitude is comparable within each member.",
        "top_features": rows[:15],
    }
    (args.output_dir / "coefficient_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
