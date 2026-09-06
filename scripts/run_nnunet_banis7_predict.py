#!/usr/bin/env python3
"""Launch the repository BANIS-7 predictor with deterministic inference settings."""

from __future__ import annotations

import os

# Must be set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import runpy
from pathlib import Path

import numpy as np
import torch


VENDOR_PREDICTOR = Path(
    "/projects/weilab/liupeng/code/vendors/nnUNet/scripts/predict_banis7.py"
)


def configure_determinism(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main() -> None:
    configure_determinism(0)
    if torch.cuda.is_available():
        print(
            f"Deterministic nnU-Net inference; PyTorch {torch.__version__}; "
            f"CUDA device: {torch.cuda.get_device_name(0)}; "
            f"capability={torch.cuda.get_device_capability(0)}",
            flush=True,
        )
    runpy.run_path(str(VENDOR_PREDICTOR), run_name="__main__")


if __name__ == "__main__":
    main()
