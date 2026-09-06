#!/usr/bin/env python3
"""Evaluate locked SDT/graph decoders on physical instance morphometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.instance_morphometry import (  # noqa: E402
    evaluate_morphometry,
    scan_morphometry_statistics,
)
from src.inference.sdt_affinity_edge_scorer import (  # noqa: E402
    EDGE_FEATURE_NAMES,
    edge_feature_matrix,
    scored_merge_lut,
)
from src.inference.sdt_affinity_graph import (  # noqa: E402
    affinity_merge_lut,
    load_volume,
    metrics_from_contingency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-audit-dir", type=Path, required=True)
    parser.add_argument("--region-mws-lut", type=Path, required=True)
    parser.add_argument("--edge-scorer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-threshold", type=float, default=0.40)
    parser.add_argument("--probability-threshold", type=float, default=0.70)
    parser.add_argument("--maximum-uncertainty", type=float, default=1.0)
    parser.add_argument("--minimum-evidence", type=int, default=8)
    parser.add_argument("--minimum-affinity-mean", type=float, default=0.40)
    parser.add_argument("--minimum-high-fraction", type=float, default=0.50)
    parser.add_argument("--block-shape", type=int, nargs=3, default=(32, 256, 256))
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    started = time.time()
    graph_config_path = args.graph_audit_dir / "config.json"
    edge_path = args.graph_audit_dir / "edge_evidence.csv"
    config = json.loads(graph_config_path.read_text())
    edges = read_csv(edge_path)
    fragments = load_volume(Path(config["fragments"]))
    ground_truth = load_volume(Path(config["ground_truth"]))
    mask_path = config.get("mask")
    mask = load_volume(Path(mask_path)) if mask_path else None
    spacing = tuple(float(value) for value in config["voxel_spacing_nm_zyx"])

    statistics = scan_morphometry_statistics(
        ground_truth,
        fragments,
        mask=mask,
        voxel_spacing_nm_zyx=spacing,
        block_shape=args.block_shape,
    )
    n_labels = max(statistics.contingency.fragment_areas, default=0) + 1
    deterministic_lut, deterministic_merges = affinity_merge_lut(
        n_labels,
        edges,
        threshold=args.graph_threshold,
        minimum_evidence=args.minimum_evidence,
        minimum_high_fraction=args.minimum_high_fraction,
    )

    scorer = joblib.load(args.edge_scorer)
    if tuple(scorer.feature_names) != EDGE_FEATURE_NAMES:
        raise RuntimeError("Edge scorer feature schema does not match the current implementation")
    features = edge_feature_matrix(edges, statistics.contingency.fragment_areas, spacing)
    probabilities, uncertainties = scorer.predict_distribution(features)
    learned_lut, learned_merges = scored_merge_lut(
        n_labels,
        edges,
        probabilities,
        uncertainties,
        probability_threshold=args.probability_threshold,
        maximum_uncertainty=args.maximum_uncertainty,
        minimum_evidence=args.minimum_evidence,
        minimum_affinity_mean=args.minimum_affinity_mean,
        minimum_high_fraction=args.minimum_high_fraction,
    )
    region_mws_lut = np.load(args.region_mws_lut)
    if len(region_mws_lut) <= max(statistics.contingency.fragment_areas, default=0):
        raise RuntimeError("Region-MWS lookup table does not cover the common fragment labels")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "banis_gr_label_lut.npy", deterministic_lut)
    np.save(args.output_dir / "banis_gr_plus_label_lut.npy", learned_lut)
    methods = (
        ("sdt_fragments", None, 0),
        ("region_mws", region_mws_lut, ""),
        ("banis_gr", deterministic_lut, deterministic_merges),
        ("banis_gr_plus", learned_lut, learned_merges),
    )
    metric_rows = []
    object_rows = []
    matched_rows = []
    for method, label_lut, accepted_merges in methods:
        morphometry = evaluate_morphometry(
            statistics,
            label_lut=label_lut,
            iou_threshold=args.iou_threshold,
        )
        instance = metrics_from_contingency(
            statistics.contingency,
            label_lut=label_lut,
            iou_threshold=args.iou_threshold,
        )
        if int(instance["tp"]) != int(morphometry.metrics["n_matched_iou50"]):
            raise RuntimeError(f"{method}: morphometry and instance matching disagree")
        metric_rows.append(
            {
                "case": config["case"],
                "method": method,
                "accepted_merges": accepted_merges,
                "instance_f1_check": instance["f1"],
                "instance_panoptic_quality_check": instance["panoptic_quality"],
                **morphometry.metrics,
            }
        )
        object_rows.extend({"case": config["case"], "method": method, **row} for row in morphometry.object_rows)
        matched_rows.extend({"case": config["case"], "method": method, **row} for row in morphometry.matched_rows)
        print(
            f"{config['case']} {method}: n={morphometry.metrics['n_pred']}, "
            f"count_abs_rel={morphometry.metrics['count_absolute_relative_error']:.4f}, "
            f"volume_W1_log={morphometry.metrics['volume_distribution_wasserstein_log']:.4f}",
            flush=True,
        )

    write_csv(args.output_dir / "method_metrics.csv", metric_rows)
    write_csv(args.output_dir / "objects.csv", object_rows)
    write_csv(args.output_dir / "matched_objects.csv", matched_rows)
    record = {
        "case": config["case"],
        "role": "locked downstream analysis; no morphometry-dependent parameter selection",
        "graph_audit_dir": str(args.graph_audit_dir),
        "region_mws_lut": str(args.region_mws_lut),
        "edge_scorer": str(args.edge_scorer),
        "input_sha256": {
            "graph_config": sha256(graph_config_path),
            "edge_evidence": sha256(edge_path),
            "region_mws_lut": sha256(args.region_mws_lut),
            "edge_scorer": sha256(args.edge_scorer),
        },
        "voxel_spacing_nm_zyx": spacing,
        "block_shape": args.block_shape,
        "iou_threshold": args.iou_threshold,
        "graph_threshold": args.graph_threshold,
        "probability_threshold": args.probability_threshold,
        "maximum_uncertainty": args.maximum_uncertainty,
        "minimum_evidence": args.minimum_evidence,
        "minimum_affinity_mean": args.minimum_affinity_mean,
        "minimum_high_fraction": args.minimum_high_fraction,
        "metrics": {
            "count": "absolute and signed (n_pred-n_true)/n_true",
            "distribution": "1-Wasserstein distance between natural-log physical volumes or elongations",
            "elongation": "sqrt(lambda_max/lambda_min) of physical-coordinate covariance plus within-voxel variance",
            "matched_errors": "absolute natural-log ratio on the same IoU>=0.5 Hungarian matches used for instance F1",
            "best_overlap": "per-GT maximum intersection/GT volume; per-pred maximum intersection/pred volume",
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")


if __name__ == "__main__":
    main()
