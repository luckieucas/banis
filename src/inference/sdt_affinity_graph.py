"""Region-graph analysis and merge-only repair for BANIS SDT fragments.

The module is deliberately independent of the training code.  It supports three
uses that are useful before committing to a new end-to-end model:

1. measure a strict merge-only oracle ceiling for an SDT watershed;
2. aggregate BANIS affinity predictions between fragment pairs; and
3. apply a deterministic affinity-threshold agglomeration baseline.

All expensive volume operations are blockwise.  Input arrays can therefore be
NumPy arrays, memory maps, or Zarr arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


Offset = Tuple[int, int, int]
Pair = Tuple[int, int]

BANIS_AFFINITY_OFFSETS: Mapping[int, Offset] = {
    0: (1, 0, 0),
    1: (0, 1, 0),
    2: (0, 0, 1),
    3: (10, 0, 0),
    4: (0, 10, 0),
    5: (0, 0, 10),
}

LOCAL_ADJACENCY_OFFSETS: Tuple[Offset, ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)


def _validate_shape(shape: Sequence[int]) -> Tuple[int, int, int]:
    shape = tuple(int(x) for x in shape)
    if len(shape) != 3 or any(x <= 0 for x in shape):
        raise ValueError(f"Expected a positive 3-D shape, got {shape}")
    return shape  # type: ignore[return-value]


def _validate_offset(offset: Sequence[int], shape: Sequence[int]) -> Offset:
    offset = tuple(int(x) for x in offset)
    if len(offset) != 3 or any(x < 0 for x in offset) or not any(offset):
        raise ValueError(f"Expected a non-zero, non-negative 3-D offset, got {offset}")
    if any(delta >= dim for delta, dim in zip(offset, shape)):
        raise ValueError(f"Offset {offset} does not fit shape {tuple(shape)}")
    return offset  # type: ignore[return-value]


def offset_slices(shape: Sequence[int], offset: Sequence[int]) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """Return aligned source/destination slices for a positive offset."""
    shape = _validate_shape(shape)
    offset = _validate_offset(offset, shape)
    source = tuple(slice(0, dim - delta) for dim, delta in zip(shape, offset))
    target = tuple(slice(delta, dim) for dim, delta in zip(shape, offset))
    return source, target


def iter_blocks(shape: Sequence[int], block_shape: Sequence[int]) -> Iterator[Tuple[slice, slice, slice]]:
    """Yield non-overlapping blocks covering ``shape`` exactly once."""
    shape = _validate_shape(shape)
    block_shape = tuple(int(x) for x in block_shape)
    if len(block_shape) != 3 or any(x <= 0 for x in block_shape):
        raise ValueError(f"Expected a positive 3-D block shape, got {block_shape}")
    for z in range(0, shape[0], block_shape[0]):
        for y in range(0, shape[1], block_shape[1]):
            for x in range(0, shape[2], block_shape[2]):
                yield (
                    slice(z, min(z + block_shape[0], shape[0])),
                    slice(y, min(y + block_shape[1], shape[1])),
                    slice(x, min(x + block_shape[2], shape[2])),
                )


def shifted_block(block: Sequence[slice], offset: Sequence[int]) -> Tuple[slice, slice, slice]:
    """Shift a concrete block by ``offset`` without changing its extent."""
    return tuple(
        slice(int(s.start) + int(delta), int(s.stop) + int(delta))
        for s, delta in zip(block, offset)
    )  # type: ignore[return-value]


def _check_matching_shapes(reference, arrays: Iterable[object]) -> Tuple[int, int, int]:
    shape = _validate_shape(reference.shape)
    for array in arrays:
        if tuple(array.shape) != shape:
            raise ValueError(f"Shape mismatch: expected {shape}, got {tuple(array.shape)}")
    return shape


def _pair_counts(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count integer pairs without assuming compact or small label IDs."""
    if first.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    pairs = np.empty((first.size, 2), dtype=np.int64)
    pairs[:, 0] = first.astype(np.int64, copy=False)
    pairs[:, 1] = second.astype(np.int64, copy=False)
    return np.unique(pairs, axis=0, return_counts=True)


def collect_candidate_pairs(
    fragments,
    offsets: Sequence[Offset] = LOCAL_ADJACENCY_OFFSETS,
    mask=None,
    block_shape: Sequence[int] = (32, 256, 256),
) -> set[Pair]:
    """Collect distinct non-background fragment pairs at the requested offsets."""
    shape = _check_matching_shapes(fragments, [] if mask is None else [mask])
    result: set[Pair] = set()
    for offset in offsets:
        offset = _validate_offset(offset, shape)
        eligible_shape = tuple(dim - delta for dim, delta in zip(shape, offset))
        for source_block in iter_blocks(eligible_shape, block_shape):
            target_block = shifted_block(source_block, offset)
            source = np.asarray(fragments[source_block])
            target = np.asarray(fragments[target_block])
            valid = (source > 0) & (target > 0) & (source != target)
            if mask is not None:
                valid &= np.asarray(mask[source_block], dtype=bool)
                valid &= np.asarray(mask[target_block], dtype=bool)
            if not np.any(valid):
                continue
            low = np.minimum(source[valid], target[valid])
            high = np.maximum(source[valid], target[valid])
            pairs = np.unique(np.column_stack((low, high)), axis=0)
            result.update((int(a), int(b)) for a, b in pairs)
    return result


@dataclass
class _RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    high_count: int = 0

    def update(self, count: int, total: float, total_sq: float, high_count: int) -> None:
        self.count += int(count)
        self.total += float(total)
        self.total_sq += float(total_sq)
        self.high_count += int(high_count)

    def as_dict(self, prefix: str = "") -> Dict[str, float | int]:
        mean = self.total / self.count if self.count else 0.0
        variance = max(self.total_sq / self.count - mean * mean, 0.0) if self.count else 0.0
        return {
            f"{prefix}count": self.count,
            f"{prefix}mean": mean,
            f"{prefix}std": variance**0.5,
            f"{prefix}high_fraction": self.high_count / self.count if self.count else 0.0,
        }


def collect_affinity_evidence(
    fragments,
    affinities: Mapping[int, object],
    offsets: Mapping[int, Offset] = BANIS_AFFINITY_OFFSETS,
    mask=None,
    block_shape: Sequence[int] = (32, 256, 256),
    positive_threshold: float = 0.5,
) -> list[Dict[str, float | int]]:
    """Aggregate affinity statistics for every cross-fragment endpoint pair.

    An affinity value at a source voxel describes the edge to the positively
    shifted target voxel, matching ``src/data/data.py::comp_affinities``.
    """
    shape = _check_matching_shapes(
        fragments,
        [*affinities.values(), *([] if mask is None else [mask])],
    )
    by_pair: Dict[Pair, Dict[int, _RunningStats]] = {}

    for channel, affinity in sorted(affinities.items()):
        if channel not in offsets:
            raise ValueError(f"No offset supplied for affinity channel {channel}")
        offset = _validate_offset(offsets[channel], shape)
        eligible_shape = tuple(dim - delta for dim, delta in zip(shape, offset))
        for source_block in iter_blocks(eligible_shape, block_shape):
            target_block = shifted_block(source_block, offset)
            source = np.asarray(fragments[source_block])
            target = np.asarray(fragments[target_block])
            values = np.asarray(affinity[source_block], dtype=np.float32)
            valid = (source > 0) & (target > 0) & (source != target) & np.isfinite(values)
            if mask is not None:
                valid &= np.asarray(mask[source_block], dtype=bool)
                valid &= np.asarray(mask[target_block], dtype=bool)
            if not np.any(valid):
                continue

            low = np.minimum(source[valid], target[valid]).astype(np.int64, copy=False)
            high = np.maximum(source[valid], target[valid]).astype(np.int64, copy=False)
            edge_values = values[valid].astype(np.float64, copy=False)
            pair_rows = np.column_stack((low, high))
            unique_pairs, inverse = np.unique(pair_rows, axis=0, return_inverse=True)
            counts = np.bincount(inverse)
            totals = np.bincount(inverse, weights=edge_values)
            totals_sq = np.bincount(inverse, weights=edge_values * edge_values)
            high_counts = np.bincount(inverse, weights=edge_values >= positive_threshold)

            for index, (left, right) in enumerate(unique_pairs):
                pair = (int(left), int(right))
                channel_stats = by_pair.setdefault(pair, {}).setdefault(channel, _RunningStats())
                channel_stats.update(
                    int(counts[index]),
                    float(totals[index]),
                    float(totals_sq[index]),
                    int(high_counts[index]),
                )

    rows: list[Dict[str, float | int]] = []
    for (left, right), channel_stats in sorted(by_pair.items()):
        pooled = _RunningStats()
        row: Dict[str, float | int] = {"fragment_a": left, "fragment_b": right}
        for channel, stats in sorted(channel_stats.items()):
            row.update(stats.as_dict(prefix=f"ch{channel}_"))
            pooled.update(stats.count, stats.total, stats.total_sq, stats.high_count)
        row.update(pooled.as_dict())
        row["n_channels"] = len(channel_stats)
        rows.append(row)
    return rows


def pool_affinity_evidence_channels(
    edge_rows: Sequence[Mapping[str, float | int | str]],
    channels: Sequence[int],
    require_short_range_support: bool = False,
) -> list[Dict[str, float | int]]:
    """Re-pool cached per-channel edge statistics for an offset ablation.

    The cached graph audit stores count, mean, standard deviation, and the
    fraction above the affinity threshold for every channel.  These sufficient
    statistics let us compare short- and long-range offsets without rescanning
    the prediction volumes.  ``require_short_range_support`` restricts the
    selected rows to local fragment adjacencies (evidence in channels 0--2),
    while still allowing all requested channels to contribute to their score.
    """
    selected_channels = tuple(sorted({int(channel) for channel in channels}))
    if not selected_channels:
        raise ValueError("At least one affinity channel is required")
    unknown = [channel for channel in selected_channels if channel not in BANIS_AFFINITY_OFFSETS]
    if unknown:
        raise ValueError(f"Unknown affinity channels: {unknown}")

    pooled_rows: list[Dict[str, float | int]] = []
    for row in edge_rows:
        short_count = sum(int(float(row.get(f"ch{channel}_count", 0) or 0)) for channel in (0, 1, 2))
        if require_short_range_support and short_count == 0:
            continue

        pooled = _RunningStats()
        present = 0
        for channel in selected_channels:
            prefix = f"ch{channel}_"
            count = int(float(row.get(f"{prefix}count", 0) or 0))
            if count <= 0:
                continue
            mean = float(row.get(f"{prefix}mean", 0) or 0)
            std = float(row.get(f"{prefix}std", 0) or 0)
            high_fraction = float(row.get(f"{prefix}high_fraction", 0) or 0)
            pooled.update(
                count,
                count * mean,
                count * (std * std + mean * mean),
                int(round(count * high_fraction)),
            )
            present += 1
        if pooled.count == 0:
            continue
        pooled_rows.append(
            {
                "fragment_a": int(row["fragment_a"]),
                "fragment_b": int(row["fragment_b"]),
                **pooled.as_dict(),
                "n_channels": present,
            }
        )
    return pooled_rows


@dataclass(frozen=True)
class FragmentAssignments:
    sizes: np.ndarray
    dominant_gt: np.ndarray
    purity: np.ndarray


@dataclass(frozen=True)
class FragmentContingency:
    """Compact sufficient statistics for evaluating any merge-only label LUT."""

    gt_areas: Mapping[int, int]
    fragment_areas: Mapping[int, int]
    overlaps: Mapping[Pair, int]
    binary_tp: int
    binary_fp: int
    binary_fn: int


def build_fragment_contingency(
    ground_truth,
    fragments,
    mask=None,
    block_shape: Sequence[int] = (32, 256, 256),
) -> FragmentContingency:
    """Scan a volume once and retain statistics needed by all merge sweeps."""
    shape = _check_matching_shapes(
        ground_truth,
        [fragments, *([] if mask is None else [mask])],
    )
    gt_areas: Dict[int, int] = {}
    fragment_areas: Dict[int, int] = {}
    overlaps: Dict[Pair, int] = {}
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

        for values, target in ((gt[gt_fg], gt_areas), (fragment[fragment_fg], fragment_areas)):
            labels, counts = np.unique(values, return_counts=True)
            for label, count in zip(labels, counts):
                label_int = int(label)
                target[label_int] = target.get(label_int, 0) + int(count)

        both = gt_fg & fragment_fg
        if np.any(both):
            pairs, counts = _pair_counts(gt[both], fragment[both])
            for pair, count in zip(pairs, counts):
                key = (int(pair[0]), int(pair[1]))
                overlaps[key] = overlaps.get(key, 0) + int(count)

    return FragmentContingency(
        gt_areas=gt_areas,
        fragment_areas=fragment_areas,
        overlaps=overlaps,
        binary_tp=binary_tp,
        binary_fp=binary_fp,
        binary_fn=binary_fn,
    )


def assignments_from_contingency(contingency: FragmentContingency) -> FragmentAssignments:
    """Derive dominant-GT assignments from a compact contingency table."""
    max_fragment = max(contingency.fragment_areas, default=0)
    sizes = np.zeros(max_fragment + 1, dtype=np.int64)
    dominant_gt = np.zeros(max_fragment + 1, dtype=np.int64)
    dominant_count = np.zeros(max_fragment + 1, dtype=np.int64)
    for fragment, count in contingency.fragment_areas.items():
        sizes[fragment] = count
    for (gt, fragment), count in contingency.overlaps.items():
        if count > dominant_count[fragment] or (
            count == dominant_count[fragment] and gt < dominant_gt[fragment]
        ):
            dominant_count[fragment] = count
            dominant_gt[fragment] = gt
    purity = np.divide(
        dominant_count,
        sizes,
        out=np.zeros_like(dominant_count, dtype=np.float64),
        where=sizes > 0,
    )
    return FragmentAssignments(sizes, dominant_gt, purity)


def assign_fragments_to_ground_truth(
    fragments,
    ground_truth,
    mask=None,
    block_shape: Sequence[int] = (32, 256, 256),
) -> FragmentAssignments:
    """Assign each fragment to its dominant GT object and report its purity."""
    contingency = build_fragment_contingency(
        ground_truth,
        fragments,
        mask=mask,
        block_shape=block_shape,
    )
    return assignments_from_contingency(contingency)


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        root = int(item)
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[item]) != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True

    def compact_lut(self) -> np.ndarray:
        roots = np.fromiter((self.find(i) for i in range(len(self.parent))), dtype=np.int64)
        roots[0] = 0
        positive = np.unique(roots[1:])
        compact = np.zeros(len(self.parent), dtype=np.uint32)
        for new_label, root in enumerate(positive, start=1):
            compact[roots == root] = new_label
        compact[0] = 0
        return compact


def oracle_merge_lut(
    n_labels: int,
    candidate_pairs: Iterable[Pair],
    assignments: FragmentAssignments,
    minimum_purity: float = 0.9,
) -> Tuple[np.ndarray, int]:
    """Build a strict merge-only oracle using dominant GT labels.

    A pair is merged only when both fragments have at least ``minimum_purity``
    and share the same non-background dominant GT instance.
    """
    if n_labels > len(assignments.sizes):
        raise ValueError("Fragment assignment arrays do not cover all labels")
    union_find = UnionFind(n_labels)
    accepted = 0
    for left, right in sorted(candidate_pairs):
        if left >= n_labels or right >= n_labels:
            raise ValueError(f"Candidate pair {(left, right)} exceeds n_labels={n_labels}")
        same_gt = assignments.dominant_gt[left] > 0 and (
            assignments.dominant_gt[left] == assignments.dominant_gt[right]
        )
        pure = assignments.purity[left] >= minimum_purity and assignments.purity[right] >= minimum_purity
        if same_gt and pure and union_find.union(left, right):
            accepted += 1
    return union_find.compact_lut(), accepted


def affinity_merge_lut(
    n_labels: int,
    edge_rows: Sequence[Mapping[str, float | int]],
    threshold: float,
    minimum_evidence: int = 1,
    minimum_high_fraction: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Agglomerate fragment pairs passing deterministic evidence thresholds."""
    union_find = UnionFind(n_labels)
    accepted = 0
    ranked = sorted(edge_rows, key=lambda row: (-float(row["mean"]), int(row["fragment_a"]), int(row["fragment_b"])))
    for row in ranked:
        if int(row["count"]) < minimum_evidence:
            continue
        if float(row["mean"]) < threshold:
            continue
        if float(row["high_fraction"]) < minimum_high_fraction:
            continue
        left, right = int(row["fragment_a"]), int(row["fragment_b"])
        if left >= n_labels or right >= n_labels:
            raise ValueError(f"Evidence pair {(left, right)} exceeds n_labels={n_labels}")
        if union_find.union(left, right):
            accepted += 1
    return union_find.compact_lut(), accepted


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-10 else 0.0


def metrics_from_contingency(
    contingency: FragmentContingency,
    label_lut: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, float | int]:
    """Evaluate a merge LUT from a precomputed fragment/GT contingency table."""
    pred_areas: Dict[int, int] = {}
    overlaps: Dict[Pair, int] = {}
    if label_lut is not None and contingency.fragment_areas:
        max_fragment = max(contingency.fragment_areas)
        if max_fragment >= len(label_lut):
            raise ValueError("label_lut does not cover all fragment labels")
    for fragment, count in contingency.fragment_areas.items():
        pred = int(label_lut[fragment]) if label_lut is not None else fragment
        if pred > 0:
            pred_areas[pred] = pred_areas.get(pred, 0) + count
    for (gt, fragment), count in contingency.overlaps.items():
        pred = int(label_lut[fragment]) if label_lut is not None else fragment
        if pred > 0:
            key = (gt, pred)
            overlaps[key] = overlaps.get(key, 0) + count

    true_labels = sorted(contingency.gt_areas)
    pred_labels = sorted(pred_areas)
    true_index = {label: i for i, label in enumerate(true_labels)}
    pred_index = {label: i for i, label in enumerate(pred_labels)}
    scores = np.zeros((len(true_labels), len(pred_labels)), dtype=np.float64)
    for (true_label, pred_label), intersection in overlaps.items():
        union = contingency.gt_areas[true_label] + pred_areas[pred_label] - intersection
        scores[true_index[true_label], pred_index[pred_label]] = intersection / union if union else 0.0

    n_true = len(true_labels)
    n_pred = len(pred_labels)
    n_matched = min(n_true, n_pred)
    if n_matched and np.any(scores >= iou_threshold):
        costs = -(scores >= iou_threshold).astype(float) - scores / (2.0 * n_matched)
        true_ind, pred_ind = linear_sum_assignment(costs)
        ok = scores[true_ind, pred_ind] >= iou_threshold
        tp = int(np.count_nonzero(ok))
        sum_matched_score = float(np.sum(scores[true_ind, pred_ind][ok]))
    else:
        tp = 0
        sum_matched_score = 0.0
    fp = n_pred - tp
    fn = n_true - tp
    precision = tp / (tp + fp) if tp else 0.0
    recall = tp / (tp + fn) if tp else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if tp else 0.0
    accuracy = tp / (tp + fp + fn) if tp else 0.0
    binary_tp = contingency.binary_tp
    binary_fp = contingency.binary_fp
    binary_fn = contingency.binary_fn
    binary_precision = binary_tp / (binary_tp + binary_fp) if binary_tp + binary_fp else 0.0
    binary_recall = binary_tp / (binary_tp + binary_fn) if binary_tp + binary_fn else 0.0
    binary_f1 = (
        2.0 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if binary_precision + binary_recall
        else 0.0
    )
    binary_accuracy = (
        binary_tp / (binary_tp + binary_fp + binary_fn)
        if binary_tp + binary_fp + binary_fn
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "panoptic_quality": _safe_divide(sum_matched_score, tp + fp / 2.0 + fn / 2.0),
        "mean_true_score": _safe_divide(sum_matched_score, n_true),
        "mean_matched_score": _safe_divide(sum_matched_score, tp),
        "n_true": n_true,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "binary_accuracy": binary_accuracy,
        "binary_tp": binary_tp,
        "binary_fp": binary_fp,
        "binary_fn": binary_fn,
    }


def instance_metrics(
    ground_truth,
    fragments,
    label_lut: Optional[np.ndarray] = None,
    mask=None,
    block_shape: Sequence[int] = (32, 256, 256),
    iou_threshold: float = 0.5,
) -> Dict[str, float | int]:
    """Compute pyconnectomics-style IoU/Hungarian instance metrics blockwise."""
    contingency = build_fragment_contingency(
        ground_truth,
        fragments,
        mask=mask,
        block_shape=block_shape,
    )
    return metrics_from_contingency(contingency, label_lut=label_lut, iou_threshold=iou_threshold)


def load_volume(path: str | Path):
    """Load a supported 3-D volume, keeping Zarr inputs lazy."""
    path = Path(path)
    lower = path.name.lower()
    if lower.endswith(".zarr") or path.is_dir():
        import zarr

        return zarr.open_array(str(path), mode="r")
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        import tifffile

        try:
            return tifffile.memmap(str(path), mode="r")
        except ValueError:
            return tifffile.imread(str(path))
    if lower.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    raise ValueError(f"Unsupported volume: {path}")
