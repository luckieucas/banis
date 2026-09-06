"""Short-range affinity connected-components baseline."""

from __future__ import annotations

import numpy as np
from numba import njit

from src.inference.inference import compute_connected_component_segmentation


@njit(cache=True)
def _remove_small_inplace(segmentation: np.ndarray, counts: np.ndarray, minimum_size: int) -> None:
    flat = segmentation.ravel()
    for index in range(flat.size):
        label = flat[index]
        if label > 0 and counts[label] < minimum_size:
            flat[index] = 0


def short_affinity_components(
    affinity_probabilities: np.ndarray,
    threshold: float,
    minimum_size: int = 200,
) -> np.ndarray:
    """Threshold three positive-direction affinities and return uint32 components."""
    probabilities = np.asarray(affinity_probabilities)
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        raise ValueError(f"Expected affinity shape (3,z,y,x), got {probabilities.shape}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if minimum_size < 0:
        raise ValueError("minimum_size must be non-negative")
    hard_affinities = np.asarray(probabilities >= threshold, dtype=np.uint8)
    segmentation = compute_connected_component_segmentation(hard_affinities)
    if minimum_size > 1 and segmentation.size:
        counts = np.bincount(segmentation.ravel())
        _remove_small_inplace(segmentation, counts, minimum_size)
    return segmentation
