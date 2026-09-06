import unittest

import numpy as np

from src.inference.sdt_affinity_graph import (
    FragmentAssignments,
    affinity_merge_lut,
    assign_fragments_to_ground_truth,
    collect_affinity_evidence,
    collect_candidate_pairs,
    instance_metrics,
    offset_slices,
    oracle_merge_lut,
    pool_affinity_evidence_channels,
)


class TestSdtAffinityGraph(unittest.TestCase):
    def test_offset_slices_align_source_and_target(self):
        source, target = offset_slices((3, 4, 5), (0, 1, 0))
        volume = np.arange(60).reshape(3, 4, 5)
        np.testing.assert_array_equal(volume[target] - volume[source], 5)

    def test_candidate_pairs_are_canonical_and_masked(self):
        fragments = np.array([[[1, 1, 2, 2], [1, 3, 3, 2]]], dtype=np.uint16)
        mask = np.ones_like(fragments, dtype=bool)
        mask[0, 1, 0] = False
        pairs = collect_candidate_pairs(
            fragments,
            offsets=((0, 0, 1), (0, 1, 0)),
            mask=mask,
            block_shape=(1, 1, 2),
        )
        self.assertEqual(pairs, {(1, 2), (1, 3), (2, 3)})

    def test_affinity_evidence_uses_value_at_positive_offset_source(self):
        fragments = np.array([[[1, 1, 2, 2, 3]]], dtype=np.uint16)
        affinity = np.array([[[0.1, 0.8, 0.2, 0.6, 0.0]]], dtype=np.float32)
        rows = collect_affinity_evidence(
            fragments,
            {2: affinity},
            offsets={2: (0, 0, 1)},
            block_shape=(1, 1, 2),
            positive_threshold=0.5,
        )
        by_pair = {(row["fragment_a"], row["fragment_b"]): row for row in rows}
        self.assertAlmostEqual(by_pair[(1, 2)]["mean"], 0.8)
        self.assertAlmostEqual(by_pair[(2, 3)]["mean"], 0.6)
        self.assertEqual(by_pair[(1, 2)]["count"], 1)
        self.assertEqual(by_pair[(1, 2)]["high_fraction"], 1.0)

    def test_oracle_merges_only_pure_fragments_with_same_gt(self):
        assignments = FragmentAssignments(
            sizes=np.array([0, 5, 5, 5]),
            dominant_gt=np.array([0, 7, 7, 8]),
            purity=np.array([0.0, 1.0, 0.95, 1.0]),
        )
        lut, accepted = oracle_merge_lut(4, {(1, 2), (2, 3)}, assignments, minimum_purity=0.9)
        self.assertEqual(accepted, 1)
        self.assertEqual(lut[1], lut[2])
        self.assertNotEqual(lut[2], lut[3])

    def test_affinity_merges_transitively(self):
        rows = [
            {"fragment_a": 1, "fragment_b": 2, "count": 9, "mean": 0.9, "high_fraction": 1.0},
            {"fragment_a": 2, "fragment_b": 3, "count": 8, "mean": 0.8, "high_fraction": 0.75},
            {"fragment_a": 3, "fragment_b": 4, "count": 20, "mean": 0.4, "high_fraction": 0.1},
        ]
        lut, accepted = affinity_merge_lut(5, rows, threshold=0.7, minimum_evidence=8, minimum_high_fraction=0.5)
        self.assertEqual(accepted, 2)
        self.assertEqual(lut[1], lut[2])
        self.assertEqual(lut[2], lut[3])
        self.assertNotEqual(lut[3], lut[4])

    def test_cached_channel_statistics_are_repoolable(self):
        rows = [
            {
                "fragment_a": "1",
                "fragment_b": "2",
                "ch0_count": "2",
                "ch0_mean": "0.5",
                "ch0_std": "0.1",
                "ch0_high_fraction": "0.5",
                "ch3_count": "3",
                "ch3_mean": "0.7",
                "ch3_std": "0.2",
                "ch3_high_fraction": "1.0",
            },
            {
                "fragment_a": "2",
                "fragment_b": "3",
                "ch3_count": "4",
                "ch3_mean": "0.8",
                "ch3_std": "0.0",
                "ch3_high_fraction": "1.0",
            },
        ]
        pooled = pool_affinity_evidence_channels(rows, (0, 3))
        self.assertEqual(len(pooled), 2)
        self.assertEqual(pooled[0]["count"], 5)
        self.assertAlmostEqual(pooled[0]["mean"], 0.62)
        self.assertAlmostEqual(pooled[0]["high_fraction"], 0.8)
        self.assertAlmostEqual(pooled[0]["std"], np.sqrt(0.0376))

        local_only = pool_affinity_evidence_channels(
            rows,
            (0, 3),
            require_short_range_support=True,
        )
        self.assertEqual([(row["fragment_a"], row["fragment_b"]) for row in local_only], [(1, 2)])

    def test_assignment_purity_and_merge_improve_exact_metrics(self):
        gt = np.array([[[1, 1, 2, 2]]], dtype=np.uint16)
        fragments = np.array([[[1, 2, 3, 3]]], dtype=np.uint16)
        assignments = assign_fragments_to_ground_truth(fragments, gt, block_shape=(1, 1, 2))
        self.assertEqual(assignments.dominant_gt.tolist(), [0, 1, 1, 2])
        np.testing.assert_allclose(assignments.purity, [0.0, 1.0, 1.0, 1.0])

        baseline = instance_metrics(gt, fragments, block_shape=(1, 1, 2))
        self.assertAlmostEqual(baseline["f1"], 0.8)
        lut, accepted = oracle_merge_lut(4, {(1, 2), (2, 3)}, assignments)
        merged = instance_metrics(gt, fragments, label_lut=lut, block_shape=(1, 1, 2))
        self.assertEqual(accepted, 1)
        self.assertAlmostEqual(merged["f1"], 1.0)
        self.assertAlmostEqual(merged["accuracy"], 1.0)
        self.assertAlmostEqual(merged["binary_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
