import unittest

import numpy as np

from src.inference.instance_morphometry import (
    aggregate_fragment_moments,
    evaluate_morphometry,
    scan_morphometry_statistics,
)


class TestInstanceMorphometry(unittest.TestCase):
    def test_exact_merge_restores_count_volume_and_elongation(self) -> None:
        ground_truth = np.ones((2, 2, 5), dtype=np.uint16)
        fragments = np.ones_like(ground_truth)
        fragments[:, :, 3:] = 2
        statistics = scan_morphometry_statistics(
            ground_truth,
            fragments,
            voxel_spacing_nm_zyx=(30.0, 8.0, 8.0),
            block_shape=(1, 2, 3),
        )

        baseline = evaluate_morphometry(statistics)
        merged = evaluate_morphometry(statistics, np.asarray([0, 1, 1], dtype=np.uint32))

        self.assertEqual(baseline.metrics["n_pred"], 2)
        self.assertEqual(baseline.metrics["count_absolute_relative_error"], 1.0)
        self.assertEqual(merged.metrics["n_pred"], 1)
        self.assertAlmostEqual(merged.metrics["count_absolute_relative_error"], 0.0)
        self.assertAlmostEqual(merged.metrics["volume_distribution_wasserstein_log"], 0.0)
        self.assertAlmostEqual(merged.metrics["elongation_distribution_wasserstein_log"], 0.0)
        self.assertAlmostEqual(merged.metrics["matched_mean_absolute_volume_log_error"], 0.0)
        self.assertAlmostEqual(merged.metrics["matched_mean_absolute_elongation_log_error"], 0.0)
        self.assertEqual(merged.metrics["n_matched_iou50"], 1)

    def test_mask_and_physical_volume_are_respected(self) -> None:
        ground_truth = np.asarray([[[1, 1, 2, 2]]], dtype=np.uint16)
        fragments = np.asarray([[[4, 4, 9, 9]]], dtype=np.uint16)
        mask = np.asarray([[[1, 1, 1, 0]]], dtype=np.uint8)
        statistics = scan_morphometry_statistics(
            ground_truth,
            fragments,
            mask=mask,
            voxel_spacing_nm_zyx=(10.0, 20.0, 30.0),
            block_shape=(1, 1, 2),
        )
        result = evaluate_morphometry(statistics)
        gt_objects = [row for row in result.object_rows if row["role"] == "ground_truth"]
        self.assertEqual([row["voxel_count"] for row in gt_objects], [2.0, 1.0])
        self.assertAlmostEqual(gt_objects[0]["volume_um3"], 2 * 10 * 20 * 30 / 1.0e9)
        self.assertEqual(result.metrics["count_absolute_relative_error"], 0.0)

    def test_sparse_lookup_table_coverage_is_checked(self) -> None:
        moments = {5: np.ones(10, dtype=np.float64)}
        with self.assertRaisesRegex(ValueError, "does not cover fragment 5"):
            aggregate_fragment_moments(moments, np.zeros(5, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
