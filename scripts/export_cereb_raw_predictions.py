#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BANIS import BANIS
from src.inference.inference import patched_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Export raw BANIS prediction channels for BANIS-ready samples.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-channels", default=7, type=int)
    parser.add_argument("--small-size", default=192, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.sample_dir.parent.parent.name
    sample = args.sample_dir.name
    key = f"{dataset}__{sample}"
    out_path = args.output_dir / f"{key}.zarr"

    if out_path.exists() and not args.overwrite:
        print(f"Raw prediction exists, skipping: {out_path}", flush=True)
        return

    data_path = args.sample_dir / "data.zarr"
    data = zarr.open(str(data_path), mode="r")
    img = data["img"]

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = BANIS.load_from_checkpoint(str(args.checkpoint))
    model.eval()
    model.cuda()
    torch.set_float32_matmul_precision("medium")

    print(f"Running raw prediction for {key}: image={img.shape} {img.dtype}", flush=True)
    pred = patched_inference(
        img,
        model=model,
        small_size=args.small_size,
        do_overlap=True,
        prediction_channels=args.prediction_channels,
        divide=255,
    )
    print(f"Prediction shape={pred.shape}, dtype={pred.dtype}", flush=True)
    zarr.array(
        pred.astype(np.float16, copy=False),
        dtype=np.float16,
        store=str(out_path),
        chunks=(1, 64, 256, 256),
        overwrite=True,
    )
    print(f"Saved raw prediction: {out_path}", flush=True)


if __name__ == "__main__":
    main()
