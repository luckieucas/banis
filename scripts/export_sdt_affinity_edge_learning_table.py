#!/usr/bin/env python3
"""Export labeled, scale-aware edge features from a completed graph audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_edge_scorer import (  # noqa: E402
    EDGE_FEATURE_NAMES,
    edge_feature_dict,
    edge_supervision,
)
from src.inference.sdt_affinity_graph import (  # noqa: E402
    assignments_from_contingency,
    build_fragment_contingency,
    load_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-purity", type=float, default=0.9)
    parser.add_argument("--block-shape", type=int, nargs=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.graph_dir / "config.json"
    evidence_path = args.graph_dir / "edge_evidence.csv"
    if not config_path.is_file() or not evidence_path.is_file():
        raise SystemExit(f"Expected config.json and edge_evidence.csv in {args.graph_dir}")
    config = json.loads(config_path.read_text())
    block_shape = tuple(args.block_shape or config.get("block_shape", (32, 256, 256)))
    fragments = load_volume(Path(config["fragments"]))
    ground_truth = load_volume(Path(config["ground_truth"]))
    mask_path = config.get("mask")
    mask = load_volume(Path(mask_path)) if mask_path else None
    contingency = build_fragment_contingency(
        ground_truth,
        fragments,
        mask=mask,
        block_shape=block_shape,
    )
    assignments = assignments_from_contingency(contingency)
    with evidence_path.open(newline="") as handle:
        edge_rows = list(csv.DictReader(handle))

    output_rows = []
    for row in edge_rows:
        supervision = edge_supervision(row, assignments, minimum_purity=args.minimum_purity)
        features = edge_feature_dict(
            row,
            contingency.fragment_areas,
            config["voxel_spacing_nm_zyx"],
        )
        output_rows.append(
            {
                "case": config["case"],
                "fragment_a": int(row["fragment_a"]),
                "fragment_b": int(row["fragment_b"]),
                **supervision,
                **{f"feature_{name}": features[name] for name in EDGE_FEATURE_NAMES},
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    else:
        args.output.write_text("")
    valid = [row for row in output_rows if row["label_valid"]]
    positive = [row for row in valid if row["target_merge"]]
    metadata = {
        "case": config["case"],
        "graph_dir": str(args.graph_dir),
        "edge_table": str(args.output),
        "minimum_purity": args.minimum_purity,
        "block_shape": block_shape,
        "voxel_spacing_nm_zyx": config["voxel_spacing_nm_zyx"],
        "feature_names": list(EDGE_FEATURE_NAMES),
        "edges": len(output_rows),
        "valid_edges": len(valid),
        "positive_edges": len(positive),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"{config['case']}: exported {len(output_rows)} edges, "
        f"{len(valid)} valid, {len(positive)} positive -> {args.output}"
    )


if __name__ == "__main__":
    main()
