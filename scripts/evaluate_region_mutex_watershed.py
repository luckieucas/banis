#!/usr/bin/env python3
"""Evaluate region-graph Mutex Watershed from cached BANIS edge evidence."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.region_mutex_watershed import region_mutex_watershed_lut  # noqa: E402
from src.inference.sdt_affinity_graph import (  # noqa: E402
    build_fragment_contingency,
    load_volume,
    metrics_from_contingency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    )
    parser.add_argument("--block-shape", type=int, nargs=3)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.graph_audit_dir / "config.json"
    evidence_path = args.graph_audit_dir / "edge_evidence.csv"
    if not config_path.is_file() or not evidence_path.is_file():
        raise FileNotFoundError(f"Missing graph audit files under {args.graph_audit_dir}")
    config = json.loads(config_path.read_text())
    with evidence_path.open(newline="") as handle:
        evidence = list(csv.DictReader(handle))
    pairs = np.asarray(
        [(int(row["fragment_a"]), int(row["fragment_b"])) for row in evidence], dtype=np.int64
    ).reshape(-1, 2)
    affinities = np.asarray([float(row["mean"]) for row in evidence], dtype=np.float32)

    fragments = load_volume(Path(config["fragments"]))
    ground_truth = load_volume(Path(config["ground_truth"]))
    mask_path = config.get("mask", "")
    mask = load_volume(Path(mask_path)) if mask_path else None
    block_shape = tuple(args.block_shape or config.get("block_shape", (32, 256, 256)))
    contingency = build_fragment_contingency(
        ground_truth, fragments, mask=mask, block_shape=block_shape
    )
    fragment_ids = sorted(contingency.fragment_areas)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for beta in args.betas:
        result = region_mutex_watershed_lut(fragment_ids, pairs, affinities, beta)
        metrics = metrics_from_contingency(contingency, label_lut=result.label_lut)
        beta_text = f"{beta:.2f}"
        np.save(args.output_dir / f"label_lut_beta_{beta_text}.npy", result.label_lut)
        row = {
            "case": config["case"],
            "method": "region_mutex_watershed",
            "beta": beta,
            "fragments": result.n_fragments,
            "clusters": result.n_clusters,
            "accepted_merges": result.n_fragments - result.n_clusters,
            "attractive_edges": result.attractive_edges,
            "mutex_edges": result.mutex_edges,
            **metrics,
        }
        rows.append(row)
        print(
            f"{config['case']}: beta={beta_text}, clusters={result.n_clusters}, "
            f"F1={float(metrics['f1']):.6f}, PQ={float(metrics['panoptic_quality']):.6f}",
            flush=True,
        )
    write_rows(args.output_dir / "metrics.csv", rows)
    try:
        affogato_version = importlib.metadata.version("affogato")
    except importlib.metadata.PackageNotFoundError:
        # Conda's compiled affogato package does not install Python dist-info.
        affogato_version = "0.4.2 (conda-forge build)"
    output_config = {
        "case": config["case"],
        "source_graph_audit": str(args.graph_audit_dir),
        "fragments": config["fragments"],
        "ground_truth": config["ground_truth"],
        "mask": mask_path,
        "betas": args.betas,
        "block_shape": block_shape,
        "edge_weight": "signed pooled mean affinity minus beta; priority is absolute difference",
        "candidate_edges": len(evidence),
        "affogato_version": affogato_version,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "config.json").write_text(json.dumps(output_config, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")


if __name__ == "__main__":
    main()
