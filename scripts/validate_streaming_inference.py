#!/usr/bin/env python3
"""Compare dense and streaming BANIS overlap-add on the same real image crop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import SimpleITK as sitk
import torch
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from BANIS import BANIS  # noqa: E402
from inference.inference import (  # noqa: E402
    patched_inference_batch,
    patched_inference_batch_to_zarr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-shape", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = sitk.GetArrayFromImage(sitk.ReadImage(str(args.input)))
    requested = tuple(int(value) for value in args.crop_shape)
    crop_shape = tuple(min(dim, size) for dim, size in zip(image.shape, requested))
    starts = tuple((dim - size) // 2 for dim, size in zip(image.shape, crop_shape))
    crop = image[tuple(slice(start, start + size) for start, size in zip(starts, crop_shape))]
    crop = crop[..., None]
    model = BANIS.load_from_checkpoint(str(args.checkpoint)).eval().cuda()

    started = time.time()
    dense = patched_inference_batch(
        crop,
        model,
        small_size=args.patch_size,
        do_overlap=True,
        prediction_channels=7,
        divide=255,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        padding_mode="edge",
        padding_position="center",
    )[:7]
    dense_seconds = time.time() - started

    stream_dir = args.output_dir / "streaming"
    started = time.time()
    paths = patched_inference_batch_to_zarr(
        crop,
        model,
        stream_dir,
        output_channel_indices=(0, 1, 2, 3, 4, 5, 6),
        small_size=args.patch_size,
        do_overlap=True,
        prediction_channels=7,
        divide=255,
        batch_size=args.batch_size,
        output_block_shape=(96, 112, 104),
        output_chunks=(64, 64, 64),
        padding_mode="edge",
        padding_position="center",
    )
    streaming = np.stack([np.asarray(zarr.open_array(str(path), mode="r")) for path in paths])
    streaming_seconds = time.time() - started

    difference = np.abs(streaming.astype(np.float32) - dense)
    dense_float16 = dense.astype(np.float16)
    affinity_float16_mismatches = int(
        np.count_nonzero((streaming[:6] > 0.5) != (dense_float16[:6] > 0.5))
    )
    sdt_foreground_float16_mismatches = int(
        np.count_nonzero((streaming[6] > 0.0) != (dense_float16[6] > 0.0))
    )
    sdt_seed_float16_mismatches = int(
        np.count_nonzero((streaming[6] > 0.5) != (dense_float16[6] > 0.5))
    )
    report = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "crop_start": list(starts),
        "crop_shape": list(crop_shape),
        "dense_seconds": dense_seconds,
        "streaming_seconds": streaming_seconds,
        "exact_after_float16": bool(np.array_equal(streaming, dense_float16)),
        "mismatched_float16_values": int(np.count_nonzero(streaming != dense_float16)),
        "maximum_absolute_difference_from_dense_float32": float(difference.max(initial=0)),
        "mean_absolute_difference_from_dense_float32": float(difference.mean()),
        "affinity_threshold_0p5_mismatches_from_dense_float16": affinity_float16_mismatches,
        "sdt_foreground_threshold_0p0_mismatches_from_dense_float16": sdt_foreground_float16_mismatches,
        "sdt_seed_threshold_0p5_mismatches_from_dense_float16": sdt_seed_float16_mismatches,
        "affinity_threshold_0p5_mismatches_from_dense_float32": int(
            np.count_nonzero((streaming[:6] > 0.5) != (dense[:6] > 0.5))
        ),
        "dense_shape": list(dense.shape),
        "streaming_shape": list(streaming.shape),
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["maximum_absolute_difference_from_dense_float32"] > 1e-3:
        raise RuntimeError("Streaming prediction differs from dense inference by more than 1e-3")
    if affinity_float16_mismatches or sdt_foreground_float16_mismatches or sdt_seed_float16_mismatches:
        raise RuntimeError("Streaming inference changes at least one decision relative to persisted dense float16")


if __name__ == "__main__":
    main()
