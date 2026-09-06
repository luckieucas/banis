#!/usr/bin/env python3
"""Compare two label volumes exactly and record a blockwise provenance audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_graph import load_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-depth", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = load_volume(args.first)
    second = load_volume(args.second)
    if tuple(first.shape) != tuple(second.shape):
        result = {
            "first": str(args.first),
            "second": str(args.second),
            "first_shape": list(first.shape),
            "second_shape": list(second.shape),
            "exactly_equal": False,
            "reason": "shape_mismatch",
        }
    else:
        first_hash = hashlib.sha256()
        second_hash = hashlib.sha256()
        mismatched_voxels = 0
        first_nonzero = 0
        second_nonzero = 0
        for z in range(0, first.shape[0], args.block_depth):
            slices = slice(z, min(z + args.block_depth, first.shape[0]))
            first_block = np.asarray(first[slices], dtype=np.uint32)
            second_block = np.asarray(second[slices], dtype=np.uint32)
            first_hash.update(first_block.tobytes(order="C"))
            second_hash.update(second_block.tobytes(order="C"))
            mismatched_voxels += int(np.count_nonzero(first_block != second_block))
            first_nonzero += int(np.count_nonzero(first_block))
            second_nonzero += int(np.count_nonzero(second_block))
        result = {
            "first": str(args.first),
            "second": str(args.second),
            "shape": list(first.shape),
            "comparison_dtype": "uint32",
            "first_sha256_canonical": first_hash.hexdigest(),
            "second_sha256_canonical": second_hash.hexdigest(),
            "first_nonzero_voxels": first_nonzero,
            "second_nonzero_voxels": second_nonzero,
            "mismatched_voxels": mismatched_voxels,
            "exactly_equal": mismatched_voxels == 0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
