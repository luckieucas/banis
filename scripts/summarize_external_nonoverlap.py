#!/usr/bin/env python3
"""Recompute external aggregates after excluding provenance-overlapping cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RUN_ROOT = Path(
    "/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1"
)
EXTERNAL_ROOT = RUN_ROOT / "external_confirmatory"
KEEP_CASES = {
    "jrc_zf-cardiac-1_recon-1_test1",
    "jrc_mus-liver_recon-1_test1",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> dict[str, float | int]:
    if {row["case"] for row in rows} != KEEP_CASES:
        raise ValueError("Expected exactly the Cardiac and Liver cases")
    result: dict[str, float | int] = {"cases": len(rows)}
    for key in ("precision", "recall", "f1", "accuracy", "panoptic_quality"):
        result[f"macro_{key}"] = sum(float(row[key]) for row in rows) / len(rows)
    for key in ("tp", "fp", "fn"):
        result[key] = sum(int(row[key]) for row in rows)
    tp, fp, fn = int(result["tp"]), int(result["fp"]), int(result["fn"])
    result["pooled_precision"] = tp / (tp + fp)
    result["pooled_recall"] = tp / (tp + fn)
    result["pooled_f1"] = 2 * tp / (2 * tp + fp + fn)
    result["pooled_accuracy"] = tp / (tp + fp + fn)
    return result


def main() -> None:
    deterministic_rows = []
    for domain in ("cardiac", "liver"):
        path = next((EXTERNAL_ROOT / domain).glob("*/graph_audit/metrics.csv"))
        deterministic_rows.extend(read_rows(path))

    methods = {}
    for method in ("baseline", "strict_local_oracle", "affinity_merge"):
        rows = [
            row
            for row in deterministic_rows
            if row["method"] == method
            and (method != "affinity_merge" or float(row["threshold"]) == 0.40)
        ]
        methods[method] = aggregate(rows)

    learned = {}
    for seed, suffix in enumerate(("", "_seed1", "_seed2")):
        path = EXTERNAL_ROOT / f"learned_gate_summary{suffix}" / "per_case_metrics.csv"
        rows = [
            row
            for row in read_rows(path)
            if row["case"] in KEEP_CASES and row["method"] == "learned_edge_scorer"
        ]
        learned[f"seed{seed}"] = aggregate(rows)

    cc_path = RUN_ROOT / "short_affinity_cc/external/summary/per_case_metrics.csv"
    cc_rows = [row for row in read_rows(cc_path) if row["case"] in KEEP_CASES]
    methods["short_affinity_cc"] = aggregate(cc_rows)

    output = {
        "included_cases": sorted(KEEP_CASES),
        "excluded_case": "jrc_mus-kidney_recon-1_test1",
        "exclusion_reason": "voxel-identical to MitoEM2 Podo-test01 ground truth",
        "methods": methods,
        "learned_edge_scorer": learned,
    }
    output_dir = EXTERNAL_ROOT / "nonoverlap_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
