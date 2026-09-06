"""Blockwise physical morphometry for merge-only instance segmentations.

The SDT, BANIS-GR, BANIS-GR+, and region-graph MWS predictions differ only by
lookup tables over a common fragment image.  This module scans the ground truth
and fragment image once, accumulates exact per-label zeroth/first/second
moments, and then evaluates any number of label lookup tables without
materializing dense relabeled volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

from src.inference.sdt_affinity_graph import FragmentContingency, iter_blocks


# count, z, y, x, zz, yy, xx, zy, zx, yx
MOMENT_WIDTH = 10


@dataclass(frozen=True)
class MorphometryStatistics:
    contingency: FragmentContingency
    gt_moments: Mapping[int, np.ndarray]
    fragment_moments: Mapping[int, np.ndarray]
    voxel_spacing_nm_zyx: tuple[float, float, float]


@dataclass(frozen=True)
class MorphometryResult:
    metrics: Mapping[str, float | int]
    object_rows: tuple[Mapping[str, float | int | str], ...]
    matched_rows: tuple[Mapping[str, float | int], ...]


def _validate_inputs(ground_truth, fragments, mask, voxel_spacing_nm_zyx: Sequence[float]):
    shape = tuple(int(value) for value in ground_truth.shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"Expected a positive 3-D volume, got {shape}")
    for array in (fragments, *(() if mask is None else (mask,))):
        if tuple(array.shape) != shape:
            raise ValueError(f"Shape mismatch: expected {shape}, got {tuple(array.shape)}")
    spacing = np.asarray(voxel_spacing_nm_zyx, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Expected three positive finite spacings, got {voxel_spacing_nm_zyx}")
    return shape, spacing


def _accumulate_moments(
    labels: np.ndarray,
    roi: np.ndarray,
    block: Sequence[slice],
    spacing: np.ndarray,
    target: dict[int, np.ndarray],
) -> None:
    selected = (labels > 0) & roi
    if not np.any(selected):
        return
    local = np.nonzero(selected)
    label_values = labels[local].astype(np.int64, copy=False)
    unique, inverse = np.unique(label_values, return_inverse=True)
    coordinates = [
        (axis_indices.astype(np.float64) + float(axis_slice.start) + 0.5) * axis_spacing
        for axis_indices, axis_slice, axis_spacing in zip(local, block, spacing)
    ]
    z, y, x = coordinates
    weights = (
        np.ones(len(label_values), dtype=np.float64),
        z,
        y,
        x,
        z * z,
        y * y,
        x * x,
        z * y,
        z * x,
        y * x,
    )
    block_moments = np.stack(
        [np.bincount(inverse, weights=weight, minlength=len(unique)) for weight in weights],
        axis=1,
    )
    for label, moment in zip(unique, block_moments):
        label_int = int(label)
        if label_int in target:
            target[label_int] += moment
        else:
            target[label_int] = moment


def _accumulate_counts(values: np.ndarray, target: dict[int, int]) -> None:
    if not len(values):
        return
    labels, counts = np.unique(values, return_counts=True)
    for label, count in zip(labels, counts):
        label_int = int(label)
        target[label_int] = target.get(label_int, 0) + int(count)


def scan_morphometry_statistics(
    ground_truth,
    fragments,
    mask=None,
    voxel_spacing_nm_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    block_shape: Sequence[int] = (32, 256, 256),
) -> MorphometryStatistics:
    """Accumulate contingency and physical shape moments in one blockwise scan."""
    shape, spacing = _validate_inputs(ground_truth, fragments, mask, voxel_spacing_nm_zyx)
    gt_areas: dict[int, int] = {}
    fragment_areas: dict[int, int] = {}
    overlaps: dict[tuple[int, int], int] = {}
    gt_moments: dict[int, np.ndarray] = {}
    fragment_moments: dict[int, np.ndarray] = {}
    binary_tp = binary_fp = binary_fn = 0

    for block in iter_blocks(shape, block_shape):
        gt = np.asarray(ground_truth[block])
        fragment = np.asarray(fragments[block])
        roi = np.ones(gt.shape, dtype=bool) if mask is None else np.asarray(mask[block], dtype=bool)
        gt_fg = (gt > 0) & roi
        fragment_fg = (fragment > 0) & roi
        binary_tp += int(np.count_nonzero(gt_fg & fragment_fg))
        binary_fp += int(np.count_nonzero(~gt_fg & fragment_fg))
        binary_fn += int(np.count_nonzero(gt_fg & ~fragment_fg))

        _accumulate_counts(gt[gt_fg], gt_areas)
        _accumulate_counts(fragment[fragment_fg], fragment_areas)
        _accumulate_moments(gt, roi, block, spacing, gt_moments)
        _accumulate_moments(fragment, roi, block, spacing, fragment_moments)

        both = gt_fg & fragment_fg
        if np.any(both):
            pairs = np.empty((int(np.count_nonzero(both)), 2), dtype=np.int64)
            pairs[:, 0] = gt[both].astype(np.int64, copy=False)
            pairs[:, 1] = fragment[both].astype(np.int64, copy=False)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            for pair, count in zip(unique_pairs, counts):
                key = (int(pair[0]), int(pair[1]))
                overlaps[key] = overlaps.get(key, 0) + int(count)

    contingency = FragmentContingency(
        gt_areas=gt_areas,
        fragment_areas=fragment_areas,
        overlaps=overlaps,
        binary_tp=binary_tp,
        binary_fp=binary_fp,
        binary_fn=binary_fn,
    )
    return MorphometryStatistics(
        contingency=contingency,
        gt_moments=gt_moments,
        fragment_moments=fragment_moments,
        voxel_spacing_nm_zyx=tuple(float(value) for value in spacing),
    )


def aggregate_fragment_moments(
    fragment_moments: Mapping[int, np.ndarray], label_lut: Optional[np.ndarray] = None
) -> dict[int, np.ndarray]:
    """Aggregate exact fragment moments under a merge-only label lookup table."""
    if label_lut is not None and fragment_moments:
        maximum = max(fragment_moments)
        if maximum >= len(label_lut):
            raise ValueError(f"label_lut length {len(label_lut)} does not cover fragment {maximum}")
    result: dict[int, np.ndarray] = {}
    for fragment, moment in fragment_moments.items():
        label = fragment if label_lut is None else int(label_lut[fragment])
        if label <= 0:
            continue
        if label in result:
            result[label] += moment
        else:
            result[label] = np.asarray(moment, dtype=np.float64).copy()
    return result


def _shape_features(moment: np.ndarray, spacing: np.ndarray) -> dict[str, float]:
    count = float(moment[0])
    if count <= 0:
        raise ValueError("A label moment must contain at least one voxel")
    centroid = moment[1:4] / count
    second = np.asarray(
        [
            [moment[4], moment[7], moment[8]],
            [moment[7], moment[5], moment[9]],
            [moment[8], moment[9], moment[6]],
        ],
        dtype=np.float64,
    ) / count
    covariance = second - np.outer(centroid, centroid)
    # Treat every voxel as a finite cuboid, rather than a zero-size point.  The
    # within-voxel variance prevents degenerate elongation for thin objects.
    covariance += np.diag(spacing * spacing / 12.0)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).eps)
    elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    return {
        "voxel_count": count,
        "volume_um3": float(count * np.prod(spacing) / 1.0e9),
        "elongation": elongation,
        "centroid_z_nm": float(centroid[0]),
        "centroid_y_nm": float(centroid[1]),
        "centroid_x_nm": float(centroid[2]),
    }


def _relabel_contingency(
    contingency: FragmentContingency, label_lut: Optional[np.ndarray]
) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    if label_lut is not None and contingency.fragment_areas:
        maximum = max(contingency.fragment_areas)
        if maximum >= len(label_lut):
            raise ValueError(f"label_lut length {len(label_lut)} does not cover fragment {maximum}")
    pred_areas: dict[int, int] = {}
    overlaps: dict[tuple[int, int], int] = {}
    for fragment, count in contingency.fragment_areas.items():
        pred = fragment if label_lut is None else int(label_lut[fragment])
        if pred > 0:
            pred_areas[pred] = pred_areas.get(pred, 0) + count
    for (true_label, fragment), count in contingency.overlaps.items():
        pred = fragment if label_lut is None else int(label_lut[fragment])
        if pred > 0:
            key = (true_label, pred)
            overlaps[key] = overlaps.get(key, 0) + count
    return pred_areas, overlaps


def evaluate_morphometry(
    statistics: MorphometryStatistics,
    label_lut: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
) -> MorphometryResult:
    """Evaluate count and physical shape recovery for one merge-only labeling."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must lie in [0, 1], got {iou_threshold}")
    spacing = np.asarray(statistics.voxel_spacing_nm_zyx, dtype=np.float64)
    pred_moments = aggregate_fragment_moments(statistics.fragment_moments, label_lut)
    true_features = {
        label: _shape_features(moment, spacing) for label, moment in statistics.gt_moments.items()
    }
    pred_features = {label: _shape_features(moment, spacing) for label, moment in pred_moments.items()}
    pred_areas, overlaps = _relabel_contingency(statistics.contingency, label_lut)

    true_labels = sorted(statistics.contingency.gt_areas)
    pred_labels = sorted(pred_areas)
    true_index = {label: index for index, label in enumerate(true_labels)}
    pred_index = {label: index for index, label in enumerate(pred_labels)}
    iou = np.zeros((len(true_labels), len(pred_labels)), dtype=np.float64)
    for (true_label, pred_label), intersection in overlaps.items():
        union = statistics.contingency.gt_areas[true_label] + pred_areas[pred_label] - intersection
        iou[true_index[true_label], pred_index[pred_label]] = intersection / union if union else 0.0

    matched: list[tuple[int, int, float, int]] = []
    n_possible = min(len(true_labels), len(pred_labels))
    if n_possible and np.any(iou >= iou_threshold):
        costs = -(iou >= iou_threshold).astype(np.float64) - iou / (2.0 * n_possible)
        true_indices, pred_indices = linear_sum_assignment(costs)
        for true_i, pred_i in zip(true_indices, pred_indices):
            score = float(iou[true_i, pred_i])
            if score >= iou_threshold:
                true_label, pred_label = true_labels[true_i], pred_labels[pred_i]
                matched.append((true_label, pred_label, score, overlaps[(true_label, pred_label)]))

    best_true_overlap = {label: 0 for label in true_labels}
    best_pred_overlap = {label: 0 for label in pred_labels}
    for (true_label, pred_label), intersection in overlaps.items():
        best_true_overlap[true_label] = max(best_true_overlap[true_label], intersection)
        best_pred_overlap[pred_label] = max(best_pred_overlap[pred_label], intersection)
    true_retained = [
        best_true_overlap[label] / statistics.contingency.gt_areas[label] for label in true_labels
    ]
    pred_purity = [best_pred_overlap[label] / pred_areas[label] for label in pred_labels]

    true_log_volume = np.log([true_features[label]["volume_um3"] for label in true_labels])
    pred_log_volume = np.log([pred_features[label]["volume_um3"] for label in pred_labels])
    true_log_elongation = np.log([true_features[label]["elongation"] for label in true_labels])
    pred_log_elongation = np.log([pred_features[label]["elongation"] for label in pred_labels])
    count_relative_error = (
        (len(pred_labels) - len(true_labels)) / len(true_labels) if true_labels else float("nan")
    )

    matched_rows = []
    for true_label, pred_label, score, intersection in matched:
        true_feature, pred_feature = true_features[true_label], pred_features[pred_label]
        volume_log_ratio = float(np.log(pred_feature["volume_um3"] / true_feature["volume_um3"]))
        elongation_log_ratio = float(
            np.log(pred_feature["elongation"] / true_feature["elongation"])
        )
        matched_rows.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "intersection_voxels": intersection,
                "iou": score,
                "true_volume_um3": true_feature["volume_um3"],
                "pred_volume_um3": pred_feature["volume_um3"],
                "absolute_volume_log_error": abs(volume_log_ratio),
                "signed_volume_log_ratio": volume_log_ratio,
                "true_elongation": true_feature["elongation"],
                "pred_elongation": pred_feature["elongation"],
                "absolute_elongation_log_error": abs(elongation_log_ratio),
                "signed_elongation_log_ratio": elongation_log_ratio,
            }
        )

    object_rows = []
    for role, features in (("ground_truth", true_features), ("prediction", pred_features)):
        for label in sorted(features):
            object_rows.append({"role": role, "label": label, **features[label]})

    volume_errors = [float(row["absolute_volume_log_error"]) for row in matched_rows]
    elongation_errors = [float(row["absolute_elongation_log_error"]) for row in matched_rows]
    metrics: dict[str, float | int] = {
        "n_true": len(true_labels),
        "n_pred": len(pred_labels),
        "n_matched_iou50": len(matched),
        "count_signed_relative_error": float(count_relative_error),
        "count_absolute_relative_error": float(abs(count_relative_error)),
        "mean_gt_best_overlap_fraction": float(np.mean(true_retained)) if true_retained else float("nan"),
        "mean_pred_best_purity": float(np.mean(pred_purity)) if pred_purity else float("nan"),
        "volume_distribution_wasserstein_log": (
            float(wasserstein_distance(true_log_volume, pred_log_volume))
            if len(true_log_volume) and len(pred_log_volume)
            else float("nan")
        ),
        "elongation_distribution_wasserstein_log": (
            float(wasserstein_distance(true_log_elongation, pred_log_elongation))
            if len(true_log_elongation) and len(pred_log_elongation)
            else float("nan")
        ),
        "matched_mean_absolute_volume_log_error": (
            float(np.mean(volume_errors)) if volume_errors else float("nan")
        ),
        "matched_median_absolute_volume_log_error": (
            float(np.median(volume_errors)) if volume_errors else float("nan")
        ),
        "matched_mean_absolute_elongation_log_error": (
            float(np.mean(elongation_errors)) if elongation_errors else float("nan")
        ),
        "matched_median_absolute_elongation_log_error": (
            float(np.median(elongation_errors)) if elongation_errors else float("nan")
        ),
    }
    return MorphometryResult(metrics, tuple(object_rows), tuple(matched_rows))
