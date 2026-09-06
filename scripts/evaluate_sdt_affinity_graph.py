#!/usr/bin/env python3
"""Evaluate merge-only SDT-fragment repair with BANIS affinities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_graph import (  # noqa: E402
    BANIS_AFFINITY_OFFSETS,
    LOCAL_ADJACENCY_OFFSETS,
    affinity_merge_lut,
    assignments_from_contingency,
    build_fragment_contingency,
    collect_affinity_evidence,
    collect_candidate_pairs,
    load_volume,
    metrics_from_contingency,
    oracle_merge_lut,
)


def parse_channel_path(value: str) -> tuple[int, Path]:
    try:
        channel_text, path_text = value.split("=", 1)
        channel = int(channel_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Affinity must have the form CHANNEL=PATH") from exc
    if channel not in BANIS_AFFINITY_OFFSETS:
        raise argparse.ArgumentTypeError(f"Affinity channel must be 0..5, got {channel}")
    path = Path(path_text)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Affinity path does not exist: {path}")
    return channel, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure an SDT merge-only oracle and affinity graph-repair baseline."
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument(
        "--affinity",
        action="append",
        default=[],
        type=parse_channel_path,
        metavar="CHANNEL=PATH",
        help="May be repeated. BANIS channels 0..2 use offset 1; 3..5 use offset 10.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    parser.add_argument("--minimum-evidence", type=int, default=8)
    parser.add_argument("--minimum-high-fraction", type=float, default=0.5)
    parser.add_argument("--oracle-minimum-purity", type=float, default=0.9)
    parser.add_argument("--affinity-positive-threshold", type=float, default=0.5)
    parser.add_argument("--block-shape", nargs=3, type=int, default=(32, 256, 256))
    parser.add_argument("--voxel-spacing-nm", nargs=3, type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    affinity_paths = dict(args.affinity)
    if len(affinity_paths) != len(args.affinity):
        raise ValueError("Each affinity channel may be supplied only once")

    fragments = load_volume(args.fragments)
    ground_truth = load_volume(args.ground_truth)
    mask = load_volume(args.mask) if args.mask else None
    affinities = {channel: load_volume(path) for channel, path in affinity_paths.items()}
    shape = tuple(int(x) for x in fragments.shape)
    print(f"{args.case}: shape={shape}, affinity_channels={sorted(affinities)}", flush=True)

    candidate_pairs = collect_candidate_pairs(
        fragments,
        offsets=LOCAL_ADJACENCY_OFFSETS,
        mask=mask,
        block_shape=args.block_shape,
    )
    print(f"{args.case}: local candidate pairs={len(candidate_pairs)}", flush=True)
    contingency = build_fragment_contingency(
        ground_truth,
        fragments,
        mask=mask,
        block_shape=args.block_shape,
    )
    max_fragment = max(contingency.fragment_areas, default=0)
    n_labels = max_fragment + 1
    assignments = assignments_from_contingency(contingency)
    baseline = metrics_from_contingency(contingency)
    print(f"{args.case}: in-ROI fragment labels={len(contingency.fragment_areas)} (max ID {max_fragment})", flush=True)
    oracle_lut, oracle_merges = oracle_merge_lut(
        n_labels,
        candidate_pairs,
        assignments,
        minimum_purity=args.oracle_minimum_purity,
    )
    oracle = metrics_from_contingency(contingency, label_lut=oracle_lut)
    np.save(args.output_dir / "oracle_label_lut.npy", oracle_lut)

    metrics_rows = [
        {"case": args.case, "method": "baseline", "threshold": "", "accepted_merges": 0, **baseline},
        {
            "case": args.case,
            "method": "strict_local_oracle",
            "threshold": "",
            "accepted_merges": oracle_merges,
            **oracle,
        },
    ]

    edge_rows: list[dict] = []
    if affinities:
        evidence = collect_affinity_evidence(
            fragments,
            affinities,
            mask=mask,
            block_shape=args.block_shape,
            positive_threshold=args.affinity_positive_threshold,
        )
        channels = sorted(affinities)
        edge_fields = ["fragment_a", "fragment_b", "count", "mean", "std", "high_fraction", "n_channels"]
        for channel in channels:
            edge_fields.extend(
                [f"ch{channel}_count", f"ch{channel}_mean", f"ch{channel}_std", f"ch{channel}_high_fraction"]
            )
        for row in evidence:
            edge_rows.append({field: row.get(field, "") for field in edge_fields})
        write_rows(args.output_dir / "edge_evidence.csv", edge_rows)
        print(f"{args.case}: affinity candidate pairs={len(evidence)}", flush=True)

        best_key = (-1.0, -1.0)
        best_lut = None
        for threshold in args.thresholds:
            lut, accepted = affinity_merge_lut(
                n_labels,
                evidence,
                threshold=threshold,
                minimum_evidence=args.minimum_evidence,
                minimum_high_fraction=args.minimum_high_fraction,
            )
            metrics = metrics_from_contingency(contingency, label_lut=lut)
            metrics_rows.append(
                {
                    "case": args.case,
                    "method": "affinity_merge",
                    "threshold": threshold,
                    "accepted_merges": accepted,
                    **metrics,
                }
            )
            key = (float(metrics["f1"]), float(metrics["accuracy"]))
            if key > best_key:
                best_key = key
                best_lut = lut
            print(
                f"{args.case}: threshold={threshold:.3f}, merges={accepted}, "
                f"F1={float(metrics['f1']):.4f}, Acc={float(metrics['accuracy']):.4f}",
                flush=True,
            )
        if best_lut is not None:
            np.save(args.output_dir / "best_affinity_label_lut_analysis_only.npy", best_lut)

    write_rows(args.output_dir / "metrics.csv", metrics_rows)
    physical_offsets_nm = {
        str(channel): [
            float(delta * spacing) for delta, spacing in zip(BANIS_AFFINITY_OFFSETS[channel], args.voxel_spacing_nm)
        ]
        for channel in sorted(affinities)
    }
    config = {
        "case": args.case,
        "shape": shape,
        "fragments": str(args.fragments),
        "ground_truth": str(args.ground_truth),
        "mask": str(args.mask) if args.mask else "",
        "affinities": {str(channel): str(path) for channel, path in affinity_paths.items()},
        "voxel_spacing_nm_zyx": args.voxel_spacing_nm,
        "physical_offsets_nm_zyx": physical_offsets_nm,
        "thresholds": args.thresholds,
        "minimum_evidence": args.minimum_evidence,
        "minimum_high_fraction": args.minimum_high_fraction,
        "oracle_minimum_purity": args.oracle_minimum_purity,
        "affinity_positive_threshold": args.affinity_positive_threshold,
        "block_shape": args.block_shape,
        "local_candidate_pairs": len(candidate_pairs),
        "affinity_candidate_pairs": len(edge_rows),
        "note": "Affinity threshold results are analysis-only unless thresholds are selected on held-out validation data.",
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")
    print(
        f"{args.case}: baseline F1={float(baseline['f1']):.4f}, "
        f"strict oracle F1={float(oracle['f1']):.4f}, elapsed={time.time() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
