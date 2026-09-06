#!/usr/bin/env python3
"""Decode an nnU-Net BANIS-7 SDT prediction with the frozen BANIS watershed."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_graph import load_volume  # noqa: E402
from src.inference.watershed_and_eval_sdt_only import (  # noqa: E402
    apply_watershed_segmentation_sdt_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--sdt-channel", type=int, default=0)
    parser.add_argument("--seed-threshold", type=float, default=0.5)
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=200)
    parser.add_argument("--edt-downsample-factor", type=int, default=2)
    parser.add_argument("--edt-parallel", type=int, default=8)
    return parser.parse_args()


def load_sdt_prediction(path: Path, channel: int) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
        channel_names = (
            [str(value) for value in archive["channel_names"]]
            if "channel_names" in archive
            else []
        )
        if prediction.ndim != 4 or not 0 <= channel < prediction.shape[0]:
            raise ValueError(
                f"Expected a channel-first 4-D prediction containing channel {channel}, "
                f"got {prediction.shape}"
            )
        sdt = prediction[channel].astype(np.float32, copy=True)
    if channel_names and channel_names[channel] != "sdt":
        raise ValueError(
            f"Selected channel {channel} is {channel_names[channel]!r}, not the SDT channel"
        )
    return sdt, channel_names


def decode_sdt(
    sdt: np.ndarray,
    valid: np.ndarray | None,
    *,
    seed_threshold: float,
    foreground_threshold: float,
    min_size: int,
    edt_downsample_factor: int,
    edt_parallel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sdt.ndim != 3:
        raise ValueError(f"Expected a 3-D SDT, got {sdt.shape}")
    if valid is None:
        valid = np.ones(sdt.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != sdt.shape:
            raise ValueError(f"Mask/SDT shape mismatch: {valid.shape} != {sdt.shape}")
    if not np.isfinite(sdt[valid]).all():
        raise ValueError("SDT contains non-finite values inside the evaluation mask")

    sdt = np.asarray(sdt, dtype=np.float32).copy()
    sdt[~valid] = 0.0
    foreground, initial, final = apply_watershed_segmentation_sdt_only(
        sdt,
        skeleton_thr=seed_threshold,
        foreground_thr=foreground_threshold,
        min_size=min_size,
        edt_downsample_factor=edt_downsample_factor,
        use_fast_edt=True,
        edt_parallel=edt_parallel,
        edt_anisotropy=None,
    )
    foreground[~valid] = 0
    initial[~valid] = 0
    final[~valid] = 0
    return foreground, initial, final


def main() -> None:
    args = parse_args()
    started = time.time()
    sdt, channel_names = load_sdt_prediction(args.prediction, args.sdt_channel)
    valid = load_volume(args.mask) > 0 if args.mask else None
    foreground, initial, final = decode_sdt(
        sdt,
        valid,
        seed_threshold=args.seed_threshold,
        foreground_threshold=args.foreground_threshold,
        min_size=args.min_size,
        edt_downsample_factor=args.edt_downsample_factor,
        edt_parallel=args.edt_parallel,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "foreground": args.output_dir / "foreground_mask.tiff",
        "initial": args.output_dir / "initial_instance_seg.tiff",
        "final": args.output_dir / "final_instance_seg.tiff",
    }
    tifffile.imwrite(outputs["foreground"], foreground.astype(np.uint8, copy=False), compression="zlib")
    tifffile.imwrite(outputs["initial"], initial.astype(np.uint32, copy=False), bigtiff=True, compression="zlib")
    tifffile.imwrite(outputs["final"], final.astype(np.uint32, copy=False), bigtiff=True, compression="zlib")
    metadata = {
        "prediction": str(args.prediction),
        "prediction_channel_names": channel_names,
        "sdt_channel": args.sdt_channel,
        "mask": str(args.mask) if args.mask else "",
        "shape": list(sdt.shape),
        "seed_threshold": args.seed_threshold,
        "foreground_threshold": args.foreground_threshold,
        "min_size": args.min_size,
        "edt_downsample_factor": args.edt_downsample_factor,
        "edt_parallel": args.edt_parallel,
        "output_instances": int(np.max(final, initial=0)),
        "elapsed_seconds": time.time() - started,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "SUCCESS").write_text("ok\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
