import unittest

import numpy as np

from src.inference.sdt_affinity_edge_scorer import (
    EDGE_FEATURE_NAMES,
    BootstrapLogisticEdgeScorer,
    edge_feature_dict,
    edge_feature_matrix,
    edge_supervision,
    scored_merge_lut,
)
from src.inference.sdt_affinity_graph import FragmentAssignments


def example_edge(left=1, right=2, mean=0.8):
    return {
        "fragment_a": left,
        "fragment_b": right,
        "count": 12,
        "mean": mean,
        "std": 0.1,
        "high_fraction": 0.75,
        "n_channels": 2,
        "ch0_count": 8,
        "ch0_mean": mean,
        "ch0_std": 0.1,
        "ch0_high_fraction": 0.75,
        "ch3_count": 4,
        "ch3_mean": mean - 0.05,
        "ch3_std": 0.12,
        "ch3_high_fraction": 0.5,
    }


class TestSdtAffinityEdgeScorer(unittest.TestCase):
    def test_feature_schema_is_finite_and_stable(self):
        row = example_edge()
        features = edge_feature_dict(row, {1: 10, 2: 40}, (30, 8, 8))
        self.assertEqual(tuple(features), EDGE_FEATURE_NAMES)
        self.assertTrue(np.all(np.isfinite(list(features.values()))))
        self.assertAlmostEqual(features["short_support_fraction"], 8 / 12)
        self.assertAlmostEqual(features["long_support_fraction"], 4 / 12)
        self.assertGreater(features["log_spacing_anisotropy"], 0)

    def test_physical_distance_depends_on_spacing(self):
        row = example_edge()
        isotropic = edge_feature_dict(row, {1: 10, 2: 20}, (16, 16, 16))
        anisotropic = edge_feature_dict(row, {1: 10, 2: 20}, (30, 8, 8))
        self.assertNotEqual(
            isotropic["log1p_support_distance_mean_nm"],
            anisotropic["log1p_support_distance_mean_nm"],
        )

    def test_supervision_marks_impure_edges_as_unsafe_negatives(self):
        assignments = FragmentAssignments(
            sizes=np.array([0, 10, 20, 30]),
            dominant_gt=np.array([0, 7, 7, 8]),
            purity=np.array([0.0, 0.95, 0.91, 0.7]),
        )
        positive = edge_supervision(example_edge(1, 2), assignments)
        ambiguous = edge_supervision(example_edge(2, 3), assignments)
        self.assertEqual((positive["label_valid"], positive["target_merge"]), (1, 1))
        self.assertEqual((ambiguous["label_valid"], ambiguous["target_merge"]), (1, 0))

    def test_bootstrap_scorer_returns_probability_and_uncertainty(self):
        rows = [example_edge(mean=value) for value in (0.1, 0.2, 0.25, 0.75, 0.8, 0.9)]
        features = edge_feature_matrix(rows, {1: 10, 2: 20}, (16, 16, 16))
        targets = np.array([0, 0, 0, 1, 1, 1])
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        scorer = BootstrapLogisticEdgeScorer(n_estimators=5, random_state=4)
        scorer.fit(features, targets, groups)
        probability, uncertainty = scorer.predict_distribution(features)
        self.assertEqual(probability.shape, (6,))
        self.assertEqual(uncertainty.shape, (6,))
        self.assertTrue(np.all((0 <= probability) & (probability <= 1)))
        self.assertTrue(np.all(uncertainty >= 0))
        self.assertGreater(probability[-1], probability[0])
        empty_probability, empty_uncertainty = scorer.predict_distribution(features[:0])
        self.assertEqual(empty_probability.shape, (0,))
        self.assertEqual(empty_uncertainty.shape, (0,))

    def test_scored_merge_respects_uncertainty(self):
        rows = [example_edge(1, 2), example_edge(2, 3)]
        lut, accepted = scored_merge_lut(
            4,
            rows,
            probabilities=[0.95, 0.9],
            uncertainties=[0.05, 0.4],
            probability_threshold=0.8,
            maximum_uncertainty=0.2,
            minimum_evidence=8,
        )
        self.assertEqual(accepted, 1)
        self.assertEqual(lut[1], lut[2])
        self.assertNotEqual(lut[2], lut[3])

    def test_scored_merge_is_anchored_by_affinity_safety_rule(self):
        rows = [example_edge(1, 2, mean=0.8), example_edge(2, 3, mean=0.3)]
        lut, accepted = scored_merge_lut(
            4,
            rows,
            probabilities=[1.0, 1.0],
            uncertainties=[0.0, 0.0],
            probability_threshold=0.0,
            maximum_uncertainty=1.0,
            minimum_evidence=8,
            minimum_affinity_mean=0.4,
            minimum_high_fraction=0.5,
        )
        self.assertEqual(accepted, 1)
        self.assertEqual(lut[1], lut[2])
        self.assertNotEqual(lut[2], lut[3])


if __name__ == "__main__":
    unittest.main()
