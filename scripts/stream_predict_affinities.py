#!/usr/bin/env python3
"""Run bounded-memory BANIS overlap-add inference and write channels to Zarr."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import resource
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
from inference.inference import patched_inference_batch_to_zarr  # noqa: E402


def load_image_data(file_path: Path, zarr_key: str = "") -> np.ndarray:
    """Load a scalar NIfTI/TIFF image or an array from a Zarr store."""
    lowered = str(file_path).lower()
    if lowered.endswith((".nii", ".nii.gz", ".tif", ".tiff")):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(file_path)))
    if lowered.endswith(".zarr") or file_path.is_dir():
        source = zarr.open(str(file_path), mode="r")
        if isinstance(source, zarr.Group):
            if not zarr_key:
                keys = list(source.array_keys())
                if len(keys) != 1:
                    raise ValueError(f"Zarr group contains arrays {keys}; provide --zarr-input-key")
                zarr_key = keys[0]
            if zarr_key not in source:
                raise KeyError(f"Zarr key {zarr_key!r} not found; available arrays: {list(source.array_keys())}")
            source = source[zarr_key]
        return np.asarray(source)
    raise ValueError(f"Unsupported input format: {file_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zarr-input-key", default="")
    parser.add_argument("--prediction-channels", type=int, default=7)
    parser.add_argument("--save-channels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--divide", type=int, default=255)
    parser.add_argument("--output-block-shape", type=int, nargs=3, default=(256, 512, 512))
    parser.add_argument("--output-chunks", type=int, nargs=3, default=(64, 256, 256))
    parser.add_argument("--padding-mode", choices=("constant", "edge", "reflect", "symmetric"), default="edge")
    parser.add_argument("--padding-position", choices=("end", "center"), default="center")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main() -> None:
    args = parse_args()
    started = time.time()
    configure_determinism(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("BANIS streaming inference requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"PyTorch {torch.__version__}; CUDA device: {torch.cuda.get_device_name(0)}; "
        f"capability={torch.cuda.get_device_capability(0)}",
        flush=True,
    )
    model = BANIS.load_from_checkpoint(str(args.checkpoint_path))
    model.eval().cuda()
    image = load_image_data(args.input_path, zarr_key=args.zarr_input_key)
    if image.ndim == 3:
        image = image[..., None]
    elif image.ndim == 4 and image.shape[0] < image.shape[-1]:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 4:
        raise ValueError(f"Expected a 3-D scalar or channel-last 4-D input, got {image.shape}")

    torch.cuda.reset_peak_memory_stats()
    output_paths = patched_inference_batch_to_zarr(
        image,
        model,
        args.output_dir,
        output_channel_indices=tuple(args.save_channels),
        small_size=args.patch_size,
        do_overlap=True,
        prediction_channels=args.prediction_channels,
        divide=args.divide,
        batch_size=args.batch_size,
        output_block_shape=tuple(args.output_block_shape),
        output_chunks=tuple(args.output_chunks),
        padding_mode=args.padding_mode,
        padding_position=args.padding_position,
    )
    elapsed = time.time() - started
    metadata = {
        "checkpoint": str(args.checkpoint_path),
        "input": str(args.input_path),
        "input_shape_channel_last": list(image.shape),
        "output_channels": args.save_channels,
        "output_paths": [str(path) for path in output_paths],
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "output_block_shape": args.output_block_shape,
        "output_chunks": args.output_chunks,
        "padding_mode": args.padding_mode,
        "padding_position": args.padding_position,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    (args.output_dir / "streaming_inference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
