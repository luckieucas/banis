import unittest

import numpy as np

from src.inference.region_mutex_watershed import region_mutex_watershed_lut


class TestRegionMutexWatershed(unittest.TestCase):
    def test_attractive_chain_merges(self) -> None:
        result = region_mutex_watershed_lut(
            [2, 5, 9],
            np.array([[2, 5], [5, 9]]),
            np.array([0.9, 0.8]),
            beta=0.5,
        )
        self.assertEqual(result.n_clusters, 1)
        self.assertEqual(len(set(result.label_lut[[2, 5, 9]])), 1)
        self.assertEqual(result.attractive_edges, 2)

    def test_mutex_edge_blocks_transitive_merge(self) -> None:
        result = region_mutex_watershed_lut(
            [1, 2, 3],
            np.array([[1, 2], [2, 3], [1, 3]]),
            np.array([0.9, 0.8, 0.0]),
            beta=0.5,
        )
        self.assertEqual(result.n_clusters, 2)
        self.assertEqual(result.label_lut[1], result.label_lut[2])
        self.assertNotEqual(result.label_lut[1], result.label_lut[3])

    def test_no_edges_preserves_all_fragments(self) -> None:
        result = region_mutex_watershed_lut(
            [4, 10], np.empty((0, 2), dtype=np.int64), np.empty(0), beta=0.5
        )
        self.assertEqual(result.n_clusters, 2)
        self.assertNotEqual(result.label_lut[4], result.label_lut[10])
        self.assertEqual(result.label_lut[0], 0)


if __name__ == "__main__":
    unittest.main()
