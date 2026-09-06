#!/usr/bin/env python3
"""Run MitoNet v1 on one NIfTI or Zarr volume."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import types

# Set before importing torch through the MitoNet modules.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import SimpleITK as sitk
import torch

from mitoem2.configs import load_config
from mitoem2.inference.mitonet_inference import MitoNetInferenceEngine
from mitoem2.models.mitonet.model import MitoNetModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


@torch.no_grad()
def deterministic_median(self: object, key: str) -> torch.Tensor:
    """Return the same odd-window median without CUDA's nondeterministic indices."""
    stacked = torch.cat([output[key] for output in self.median_queue], dim=0)
    sorted_values = torch.sort(stacked, dim=0, stable=True).values
    return sorted_values[self.mid_idx : self.mid_idx + 1]


def replace_cuda_median(engine: MitoNetInferenceEngine) -> None:
    # MitoNetInferenceEngine -> Engine3d -> PanopticDeepLabRenderEngine3d.
    render_engine = engine.engine.engine
    render_engine.get_median = types.MethodType(deterministic_median, render_engine)


def load_volume(path: Path) -> tuple[np.ndarray, sitk.Image | None]:
    """Load a 3-D NIfTI or a Zarr array with an optional singleton channel."""
    if path.is_dir():
        import zarr

        image = np.asarray(zarr.open(str(path), mode="r"))
        image_itk = None
    else:
        image_itk = sitk.ReadImage(str(path))
        image = sitk.GetArrayFromImage(image_itk)
    if image.ndim == 4 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D input (or trailing singleton channel), got {image.shape}")
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError(f"MitoNet expects an integer input image, got {image.dtype}")
    minimum = int(np.min(image, initial=0))
    maximum = int(np.max(image, initial=0))
    if minimum >= 0 and maximum <= np.iinfo(np.uint8).max and image.dtype != np.uint8:
        original_dtype = image.dtype
        image = image.astype(np.uint8, copy=False)
        print(
            f"Losslessly canonicalized {original_dtype} input with range "
            f"[{minimum}, {maximum}] to uint8",
            flush=True,
        )
    return image, image_itk


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed)
    print(f"Configured deterministic MitoNet inference with seed {args.seed}", flush=True)
    if torch.cuda.is_available():
        print(
            f"PyTorch {torch.__version__}; CUDA device: {torch.cuda.get_device_name(0)}; "
            f"capability={torch.cuda.get_device_capability(0)}",
            flush=True,
        )
    config = load_config(args.config, method="mitonet")
    image, image_itk = load_volume(args.input)
    model = MitoNetModel(config_path=Path(config.model.config_path))
    if config.model.checkpoint:
        model.load_weights(Path(config.model.checkpoint))
    engine = MitoNetInferenceEngine(model=model, config=config.inference.__dict__)
    replace_cuda_median(engine)
    segmentation = engine.predict(image)
    if tuple(segmentation.shape) != tuple(image.shape):
        raise ValueError(f"MitoNet changed the volume shape: {segmentation.shape} != {image.shape}")
    if np.max(segmentation, initial=0) > np.iinfo(np.uint16).max:
        raise ValueError("MitoNet instance IDs exceed uint16")
    output_itk = sitk.GetImageFromArray(np.asarray(segmentation, dtype=np.uint16))
    if image_itk is not None:
        output_itk.CopyInformation(image_itk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_itk, str(args.output), useCompression=True)
    print(f"Saved {args.output} with shape {segmentation.shape}", flush=True)


if __name__ == "__main__":
    main()
