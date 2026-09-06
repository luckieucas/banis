"""Learned, scale-aware scoring for BANIS SDT-fragment graph edges.

The deterministic graph-repair baseline in :mod:`sdt_affinity_graph` is kept
unchanged.  This module converts its saved edge statistics into an explicit
feature representation and provides a volume-bootstrap logistic ensemble.  The
ensemble mean is a merge probability; disagreement between members is an
epistemic-uncertainty estimate that can be used to abstain from risky merges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from src.inference.sdt_affinity_graph import (
    BANIS_AFFINITY_OFFSETS,
    FragmentAssignments,
    UnionFind,
)


EDGE_CHANNELS = tuple(sorted(BANIS_AFFINITY_OFFSETS))


def edge_feature_names() -> tuple[str, ...]:
    """Return the stable ordered feature schema used by learned scorers."""
    names = [
        "pooled_log1p_count",
        "pooled_mean",
        "pooled_std",
        "pooled_high_fraction",
        "channel_coverage",
        "short_support_fraction",
        "long_support_fraction",
        "log1p_support_distance_mean_nm",
        "log1p_support_distance_std_nm",
        "log1p_spacing_z_nm",
        "log1p_spacing_y_nm",
        "log1p_spacing_x_nm",
        "log_spacing_anisotropy",
        "log1p_fragment_size_small",
        "log1p_fragment_size_large",
        "abs_log_fragment_size_ratio",
    ]
    for channel in EDGE_CHANNELS:
        names.extend(
            [
                f"ch{channel}_present",
                f"ch{channel}_log1p_count",
                f"ch{channel}_support_fraction",
                f"ch{channel}_mean",
                f"ch{channel}_std",
                f"ch{channel}_high_fraction",
            ]
        )
    return tuple(names)


EDGE_FEATURE_NAMES = edge_feature_names()


def _finite_float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    value = float(value)
    return value if np.isfinite(value) else float(default)


def _fragment_size(fragment_sizes: Sequence[int] | Mapping[int, int], label: int) -> int:
    if isinstance(fragment_sizes, Mapping):
        return max(int(fragment_sizes.get(label, 0)), 0)
    return max(int(fragment_sizes[label]) if 0 <= label < len(fragment_sizes) else 0, 0)


def edge_feature_dict(
    row: Mapping[str, object],
    fragment_sizes: Sequence[int] | Mapping[int, int],
    voxel_spacing_nm: Sequence[float],
) -> dict[str, float]:
    """Convert one raw edge-evidence row into scale-aware numeric features.

    Missing channel observations are represented by zero values plus an
    explicit ``present`` indicator.  Physical-distance features are support
    weighted, so a ten-voxel edge has a different meaning for isotropic and
    anisotropic data.
    """
    spacing = np.asarray(voxel_spacing_nm, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"voxel_spacing_nm must contain three positive values, got {voxel_spacing_nm}")

    left = int(row["fragment_a"])
    right = int(row["fragment_b"])
    if left <= 0 or right <= 0 or left == right:
        raise ValueError(f"Expected two distinct positive fragment IDs, got {(left, right)}")

    pooled_count = max(_finite_float(row, "count"), 0.0)
    channel_counts = np.asarray(
        [max(_finite_float(row, f"ch{channel}_count"), 0.0) for channel in EDGE_CHANNELS],
        dtype=np.float64,
    )
    counted_support = float(channel_counts.sum())
    support_denominator = counted_support if counted_support > 0 else max(pooled_count, 1.0)
    support_fractions = channel_counts / support_denominator

    physical_distances = np.asarray(
        [
            np.linalg.norm(np.asarray(BANIS_AFFINITY_OFFSETS[channel], dtype=np.float64) * spacing)
            for channel in EDGE_CHANNELS
        ],
        dtype=np.float64,
    )
    if counted_support > 0:
        distance_mean = float(np.average(physical_distances, weights=channel_counts))
        distance_variance = float(
            np.average((physical_distances - distance_mean) ** 2, weights=channel_counts)
        )
        distance_std = distance_variance**0.5
    else:
        distance_mean = distance_std = 0.0

    left_size = _fragment_size(fragment_sizes, left)
    right_size = _fragment_size(fragment_sizes, right)
    small_size, large_size = sorted((left_size, right_size))
    log_left = np.log1p(left_size)
    log_right = np.log1p(right_size)

    result = {
        "pooled_log1p_count": float(np.log1p(pooled_count)),
        "pooled_mean": _finite_float(row, "mean"),
        "pooled_std": _finite_float(row, "std"),
        "pooled_high_fraction": _finite_float(row, "high_fraction"),
        "channel_coverage": _finite_float(row, "n_channels") / len(EDGE_CHANNELS),
        "short_support_fraction": float(support_fractions[:3].sum()),
        "long_support_fraction": float(support_fractions[3:].sum()),
        "log1p_support_distance_mean_nm": float(np.log1p(distance_mean)),
        "log1p_support_distance_std_nm": float(np.log1p(distance_std)),
        "log1p_spacing_z_nm": float(np.log1p(spacing[0])),
        "log1p_spacing_y_nm": float(np.log1p(spacing[1])),
        "log1p_spacing_x_nm": float(np.log1p(spacing[2])),
        "log_spacing_anisotropy": float(np.log(spacing.max() / spacing.min())),
        "log1p_fragment_size_small": float(np.log1p(small_size)),
        "log1p_fragment_size_large": float(np.log1p(large_size)),
        "abs_log_fragment_size_ratio": float(abs(log_left - log_right)),
    }
    for index, channel in enumerate(EDGE_CHANNELS):
        count = channel_counts[index]
        result.update(
            {
                f"ch{channel}_present": float(count > 0),
                f"ch{channel}_log1p_count": float(np.log1p(count)),
                f"ch{channel}_support_fraction": float(support_fractions[index]),
                f"ch{channel}_mean": _finite_float(row, f"ch{channel}_mean"),
                f"ch{channel}_std": _finite_float(row, f"ch{channel}_std"),
                f"ch{channel}_high_fraction": _finite_float(row, f"ch{channel}_high_fraction"),
            }
        )
    if tuple(result) != EDGE_FEATURE_NAMES:
        raise RuntimeError("Internal edge-feature order does not match EDGE_FEATURE_NAMES")
    return result


def edge_feature_matrix(
    rows: Sequence[Mapping[str, object]],
    fragment_sizes: Sequence[int] | Mapping[int, int],
    voxel_spacing_nm: Sequence[float],
) -> np.ndarray:
    """Return an ``n_edges x n_features`` float32 matrix."""
    matrix = np.empty((len(rows), len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    for index, row in enumerate(rows):
        features = edge_feature_dict(row, fragment_sizes, voxel_spacing_nm)
        matrix[index] = [features[name] for name in EDGE_FEATURE_NAMES]
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Edge feature matrix contains non-finite values")
    return matrix


def edge_supervision(
    row: Mapping[str, object],
    assignments: FragmentAssignments,
    minimum_purity: float = 0.9,
) -> dict[str, float | int]:
    """Label whether an edge is safe under the strict merge-only oracle.

    Every observed candidate edge is a valid decision example.  A positive
    label requires two pure fragments with the same non-background dominant
    instance; impure/background-dominant endpoints are explicit unsafe
    negatives.  Including these negatives is essential because the scorer is
    applied to every candidate edge at inference time.
    """
    left, right = int(row["fragment_a"]), int(row["fragment_b"])
    n = len(assignments.sizes)
    if left <= 0 or right <= 0 or left >= n or right >= n or left == right:
        raise ValueError(f"Fragment pair {(left, right)} is outside assignment arrays of length {n}")
    purity_left = float(assignments.purity[left])
    purity_right = float(assignments.purity[right])
    gt_left = int(assignments.dominant_gt[left])
    gt_right = int(assignments.dominant_gt[right])
    safe_merge = (
        purity_left >= minimum_purity
        and purity_right >= minimum_purity
        and gt_left > 0
        and gt_right > 0
        and gt_left == gt_right
    )
    return {
        "label_valid": 1,
        "target_merge": int(safe_merge),
        "fragment_a_purity": purity_left,
        "fragment_b_purity": purity_right,
    }


@dataclass
class BootstrapLogisticEdgeScorer:
    """Volume-bootstrap ensemble for merge probability and uncertainty."""

    n_estimators: int = 25
    regularization_c: float = 1.0
    random_state: int = 0
    feature_names: tuple[str, ...] = EDGE_FEATURE_NAMES
    estimators_: list[object] = field(default_factory=list, init=False)

    def fit(
        self,
        features: np.ndarray,
        targets: Sequence[int],
        groups: Sequence[object] | None = None,
    ) -> "BootstrapLogisticEdgeScorer":
        """Fit ensemble members using bootstrap resampling by volume."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int8)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected feature shape (n, {len(self.feature_names)}), got {features.shape}"
            )
        if len(features) != len(targets) or len(targets) == 0:
            raise ValueError("Features and targets must have the same non-zero length")
        if not np.all(np.isfinite(features)):
            raise ValueError("Training features contain non-finite values")
        if set(np.unique(targets)) != {0, 1}:
            raise ValueError("Training targets must contain both binary classes")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")

        if groups is None:
            groups = np.arange(len(targets))
        groups = np.asarray(groups)
        if len(groups) != len(targets):
            raise ValueError("groups must have one entry per target")
        unique_groups = np.unique(groups)
        group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        max_attempts = max(20, self.n_estimators * 20)
        attempts = 0
        while len(self.estimators_) < self.n_estimators and attempts < max_attempts:
            attempts += 1
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sample_indices = np.concatenate([group_indices[group] for group in sampled_groups])
            sample_targets = targets[sample_indices]
            if len(np.unique(sample_targets)) < 2:
                continue
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=self.regularization_c,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=self.random_state + len(self.estimators_),
                ),
            )
            estimator.fit(features[sample_indices], sample_targets)
            self.estimators_.append(estimator)
        if len(self.estimators_) != self.n_estimators:
            raise RuntimeError(
                f"Could fit only {len(self.estimators_)} of {self.n_estimators} bootstrap estimators"
            )
        return self

    def predict_distribution(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ensemble mean probability and standard deviation."""
        if not self.estimators_:
            raise RuntimeError("Scorer has not been fitted")
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected feature shape (n, {len(self.feature_names)}), got {features.shape}"
            )
        # Empty predictions or a single fragment produce no candidate edges.
        # They are valid negative evaluation outcomes, not sklearn input errors.
        if features.shape[0] == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        probabilities = np.vstack(
            [estimator.predict_proba(features)[:, 1] for estimator in self.estimators_]
        )
        return probabilities.mean(axis=0), probabilities.std(axis=0)


def scored_merge_lut(
    n_labels: int,
    edge_rows: Sequence[Mapping[str, object]],
    probabilities: Sequence[float],
    uncertainties: Sequence[float] | None = None,
    probability_threshold: float = 0.5,
    maximum_uncertainty: float = 1.0,
    minimum_evidence: int = 1,
    minimum_affinity_mean: float = 0.0,
    minimum_high_fraction: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Agglomerate affinity-qualified edges with learned selective gates."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(edge_rows),):
        raise ValueError("probabilities must have one value per edge row")
    if uncertainties is None:
        uncertainties = np.zeros(len(edge_rows), dtype=np.float64)
    uncertainties = np.asarray(uncertainties, dtype=np.float64)
    if uncertainties.shape != probabilities.shape:
        raise ValueError("uncertainties must have one value per edge row")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(uncertainties)):
        raise ValueError("Scores must be finite")

    order = sorted(
        range(len(edge_rows)),
        key=lambda index: (
            -float(probabilities[index]),
            float(uncertainties[index]),
            int(edge_rows[index]["fragment_a"]),
            int(edge_rows[index]["fragment_b"]),
        ),
    )
    union_find = UnionFind(n_labels)
    accepted = 0
    for index in order:
        row = edge_rows[index]
        if _finite_float(row, "count") < minimum_evidence:
            continue
        if _finite_float(row, "mean") < minimum_affinity_mean:
            continue
        if _finite_float(row, "high_fraction") < minimum_high_fraction:
            continue
        if probabilities[index] < probability_threshold:
            continue
        if uncertainties[index] > maximum_uncertainty:
            continue
        left, right = int(row["fragment_a"]), int(row["fragment_b"])
        if left <= 0 or right <= 0 or left >= n_labels or right >= n_labels:
            raise ValueError(f"Evidence pair {(left, right)} exceeds n_labels={n_labels}")
        if union_find.union(left, right):
            accepted += 1
    return union_find.compact_lut(), accepted
