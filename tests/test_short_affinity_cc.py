import numpy as np
import pytest

from src.inference.short_affinity_cc import short_affinity_components


def test_positive_direction_edges_form_expected_components():
    affinities = np.zeros((3, 2, 2, 3), dtype=np.float32)
    affinities[2, 0, 0, 0] = 0.9  # (0,0,0) -> (0,0,1)
    affinities[1, 1, 0, 1] = 0.8  # (1,0,1) -> (1,1,1)
    segmentation = short_affinity_components(affinities, threshold=0.5, minimum_size=0)
    assert segmentation[0, 0, 0] == segmentation[0, 0, 1] > 0
    assert segmentation[1, 0, 1] == segmentation[1, 1, 1] > 0
    assert segmentation[0, 0, 0] != segmentation[1, 0, 1]
    assert segmentation[0, 1, 2] == 0


def test_threshold_and_small_component_filtering():
    affinities = np.zeros((3, 1, 1, 4), dtype=np.float32)
    affinities[2, 0, 0, 0] = 0.9
    affinities[2, 0, 0, 2] = 0.6
    kept = short_affinity_components(affinities, threshold=0.5, minimum_size=2)
    assert np.count_nonzero(kept) == 4
    removed = short_affinity_components(affinities, threshold=0.5, minimum_size=3)
    assert np.count_nonzero(removed) == 0
    stricter = short_affinity_components(affinities, threshold=0.7, minimum_size=2)
    assert np.count_nonzero(stricter) == 2


def test_invalid_affinity_input_is_rejected():
    with pytest.raises(ValueError):
        short_affinity_components(np.zeros((2, 3, 3, 3)), 0.5)
    with pytest.raises(ValueError):
        short_affinity_components(np.zeros((3, 3, 3, 3)), 1.1)
