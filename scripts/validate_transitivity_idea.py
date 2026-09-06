#!/usr/bin/env python3
"""Offline validation for transitivity/cycle regularization of 3-D affinities.

The BANIS model predicts three nearest-neighbour affinities followed by three
long-range affinities.  For every pair of spatial axes, those predictions form
rectangular four-edge cycles.  A valid instance partition cannot have exactly
three "join" edges in such a cycle: the fourth edge must also join the same
instance.

This script measures more than the raw violation rate.  With ground truth it
also asks the decision-critical question: does the gradient implied by the
cycle hinge point in the same direction as affinity supervision?  A cycle loss
can identify an inconsistent set of edges, but it cannot know which edge is
wrong.  That ambiguity is the main failure mode to check before training.

The implementation is crop- and stride-based so it can audit the very large
Zarr predictions produced by BANIS without materialising a full volume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import zarr


AXIS_NAMES = ("z", "y", "x")


@dataclass(frozen=True)
class EdgeType:
    channel: int
    axis: int
    distance: int


@dataclass(frozen=True)
class RectangleType:
    first: EdgeType
    second: EdgeType

    @property
    def name(self) -> str:
        a = self.first
        b = self.second
        return f"{AXIS_NAMES[a.axis]}{a.distance}x{AXIS_NAMES[b.axis]}{b.distance}"

    @property
    def scale(self) -> str:
        return f"{self.first.distance}x{self.second.distance}"


def parse_crop(text: str, shape: Sequence[int]) -> tuple[slice, slice, slice]:
    """Parse ``z0:z1,y0:y1,x0:x1`` and validate it against a volume shape."""
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Crop must contain three comma-separated slices, got {text!r}"
        )
    result: list[slice] = []
    for axis, (part, dim) in enumerate(zip(parts, shape)):
        bounds = part.split(":")
        if len(bounds) != 2 or not bounds[0] or not bounds[1]:
            raise argparse.ArgumentTypeError(
                f"Crop slice must be an explicit start:stop pair, got {part!r}"
            )
        start, stop = map(int, bounds)
        if not (0 <= start < stop <= dim):
            raise argparse.ArgumentTypeError(
                f"Crop on axis {axis} is outside [0, {dim}): {start}:{stop}"
            )
        result.append(slice(start, stop))
    return tuple(result)  # type: ignore[return-value]


def crop_to_text(crop: Sequence[slice]) -> str:
    return ",".join(f"{s.start}:{s.stop}" for s in crop)


def shifted_slice(
    valid_shape: Sequence[int], stride: int, axis: int | None = None, offset: int = 0
) -> tuple[slice, slice, slice]:
    starts = [0, 0, 0]
    if axis is not None:
        starts[axis] = offset
    return tuple(
        slice(start, start + valid, stride)
        for start, valid in zip(starts, valid_shape)
    )  # type: ignore[return-value]


def shifted_two_axes_slice(
    valid_shape: Sequence[int],
    stride: int,
    axis_a: int,
    offset_a: int,
    axis_b: int,
    offset_b: int,
) -> tuple[slice, slice, slice]:
    starts = [0, 0, 0]
    starts[axis_a] = offset_a
    starts[axis_b] = offset_b
    return tuple(
        slice(start, start + valid, stride)
        for start, valid in zip(starts, valid_shape)
    )  # type: ignore[return-value]


def same_foreground_instance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a > 0) & (a == b)


def rectangle_edges(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    rectangle: RectangleType,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predicted edges, GT edges, and GT-touch flags for one rectangle.

    Edge order follows the cycle ``00 -> 10 -> 11 -> 01 -> 00``.
    Returned edge arrays have shape ``(4, n_cycles)``.
    """
    a, b = rectangle.first, rectangle.second
    if a.axis == b.axis:
        raise ValueError("A rectangle requires two distinct axes")

    valid_shape = list(ground_truth.shape)
    valid_shape[a.axis] -= a.distance
    valid_shape[b.axis] -= b.distance
    if min(valid_shape) <= 0:
        raise ValueError(
            f"Crop {ground_truth.shape} is too small for rectangle {rectangle.name}"
        )

    s00 = shifted_slice(valid_shape, stride)
    s10 = shifted_slice(valid_shape, stride, a.axis, a.distance)
    s01 = shifted_slice(valid_shape, stride, b.axis, b.distance)
    s11 = shifted_two_axes_slice(
        valid_shape, stride, a.axis, a.distance, b.axis, b.distance
    )

    pred_edges = np.stack(
        (
            prediction[a.channel][s00],
            prediction[b.channel][s10],
            prediction[a.channel][s01],
            prediction[b.channel][s00],
        ),
        axis=0,
    ).reshape(4, -1)

    g00 = ground_truth[s00]
    g10 = ground_truth[s10]
    g01 = ground_truth[s01]
    g11 = ground_truth[s11]
    gt_edges = np.stack(
        (
            same_foreground_instance(g00, g10),
            same_foreground_instance(g10, g11),
            same_foreground_instance(g01, g11),
            same_foreground_instance(g00, g01),
        ),
        axis=0,
    ).reshape(4, -1)
    gt_touch = ((g00 > 0) | (g10 > 0) | (g01 > 0) | (g11 > 0)).reshape(-1)
    return pred_edges.astype(np.float32, copy=False), gt_edges, gt_touch


def soft_cycle_hinges(edges: np.ndarray) -> np.ndarray:
    """Lukasiewicz implications for each possible conclusion edge.

    For target edge t and the other three edges e, the implication penalty is
    ``relu(sum(e) - 2 - t)``.  Shape is preserved as ``(4, n_cycles)``.
    """
    total = edges.sum(axis=0, keepdims=True)
    return np.maximum(total - 2.0 * edges - 2.0, 0.0)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC AUC without depending on scikit-learn."""
    labels = labels.astype(bool, copy=False)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def new_counts() -> dict[str, float]:
    keys = (
        "cycles",
        "gt_touch_cycles",
        "pred_active_cycles",
        "relevant_cycles",
        "error_cycles_relevant",
        "edge_occurrences_relevant",
        "edge_errors_relevant",
        "violations",
        "violations_relevant",
        "violation_edge_occurrences",
        "violation_edge_errors",
        "violation_split_involved",
        "violation_merge_involved",
        "violation_split_only",
        "safe_lone_cut_repairs",
        "binary_gradient_correct_votes",
        "binary_gradient_votes",
        "soft_active_cycles_relevant",
        "soft_active_implications_relevant",
        "soft_gradient_correct_votes",
        "soft_gradient_votes",
        "soft_score_sum_relevant",
    )
    return {key: 0.0 for key in keys}


def add_counts(target: dict[str, float], source: dict[str, float]) -> None:
    for key in target:
        target[key] += source[key]


def compute_counts(
    edges: np.ndarray,
    gt_edges: np.ndarray,
    gt_touch: np.ndarray,
    threshold: float,
    hinges: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    pred_edges = edges > threshold
    pred_active = pred_edges.any(axis=0)
    relevant = gt_touch | pred_active
    edge_errors = pred_edges != gt_edges
    any_error = edge_errors.any(axis=0)
    join_count = pred_edges.sum(axis=0)
    violation = join_count == 3
    violation_relevant = violation & relevant

    soft_scores = hinges.max(axis=0)
    active_hinges = hinges > 0
    soft_active_cycle = active_hinges.any(axis=0)
    active_relevant = active_hinges & relevant[None]

    result = new_counts()
    result["cycles"] = float(relevant.size)
    result["gt_touch_cycles"] = float(gt_touch.sum())
    result["pred_active_cycles"] = float(pred_active.sum())
    result["relevant_cycles"] = float(relevant.sum())
    result["error_cycles_relevant"] = float((any_error & relevant).sum())
    result["edge_occurrences_relevant"] = float(4 * relevant.sum())
    result["edge_errors_relevant"] = float(edge_errors[:, relevant].sum())
    result["violations"] = float(violation.sum())
    result["violations_relevant"] = float(violation_relevant.sum())
    result["violation_edge_occurrences"] = float(4 * violation_relevant.sum())
    result["violation_edge_errors"] = float(edge_errors[:, violation_relevant].sum())
    result["violation_split_involved"] = float(
        ((~pred_edges & gt_edges).any(axis=0) & violation_relevant).sum()
    )
    result["violation_merge_involved"] = float(
        ((pred_edges & ~gt_edges).any(axis=0) & violation_relevant).sum()
    )
    result["violation_split_only"] = float(
        (gt_edges.all(axis=0) & violation_relevant).sum()
    )

    if violation_relevant.any():
        vr_pred = pred_edges[:, violation_relevant]
        vr_gt = gt_edges[:, violation_relevant]
        lone_cut_index = np.argmin(vr_pred, axis=0)
        lone_cut_gt = np.take_along_axis(
            vr_gt, lone_cut_index[None], axis=0
        ).reshape(-1)
        result["safe_lone_cut_repairs"] = float(lone_cut_gt.sum())

        # The active hard implication increases the lone-cut target and
        # decreases the other three antecedents.  Count how often those four
        # directions agree with GT affinity supervision.
        gt_cut_count = (~vr_gt).sum(axis=0)
        correct_votes = lone_cut_gt.astype(np.int8) + (
            gt_cut_count - (~lone_cut_gt).astype(np.int8)
        )
        result["binary_gradient_correct_votes"] = float(correct_votes.sum())
        result["binary_gradient_votes"] = float(4 * lone_cut_gt.size)

    result["soft_active_cycles_relevant"] = float(
        (soft_active_cycle & relevant).sum()
    )
    result["soft_active_implications_relevant"] = float(active_relevant.sum())
    result["soft_score_sum_relevant"] = float(soft_scores[relevant].sum())

    # For each active implication: increasing its target is correct iff target
    # GT is join; decreasing each antecedent is correct iff its GT is cut.
    active_targets = np.argwhere(active_relevant)
    if active_targets.size:
        target_ids = active_targets[:, 0]
        cycle_ids = active_targets[:, 1]
        target_gt = gt_edges[target_ids, cycle_ids]
        gt_cut_per_cycle = (~gt_edges[:, cycle_ids]).sum(axis=0)
        correct_votes = target_gt.astype(np.int8) + (
            gt_cut_per_cycle - (~target_gt).astype(np.int8)
        )
        result["soft_gradient_correct_votes"] = float(correct_votes.sum())
        result["soft_gradient_votes"] = float(4 * target_gt.size)

    return result, soft_scores[relevant], any_error[relevant]


def finalize_counts(counts: dict[str, float]) -> dict[str, float]:
    edge_error_rate = safe_div(
        counts["edge_errors_relevant"], counts["edge_occurrences_relevant"]
    )
    violation_edge_error_rate = safe_div(
        counts["violation_edge_errors"], counts["violation_edge_occurrences"]
    )
    return {
        "cycles": int(counts["cycles"]),
        "relevant_cycles": int(counts["relevant_cycles"]),
        "gt_touch_fraction": safe_div(counts["gt_touch_cycles"], counts["cycles"]),
        "pred_active_fraction": safe_div(
            counts["pred_active_cycles"], counts["cycles"]
        ),
        "cycle_error_rate_relevant": safe_div(
            counts["error_cycles_relevant"], counts["relevant_cycles"]
        ),
        "edge_error_rate_relevant": edge_error_rate,
        "violation_count": int(counts["violations_relevant"]),
        "violation_rate_all": safe_div(counts["violations"], counts["cycles"]),
        "violation_rate_relevant": safe_div(
            counts["violations_relevant"], counts["relevant_cycles"]
        ),
        "edge_error_rate_in_violations": violation_edge_error_rate,
        "edge_error_enrichment": safe_div(
            violation_edge_error_rate, edge_error_rate
        ),
        "violation_error_coverage_occurrence": safe_div(
            counts["violation_edge_errors"], counts["edge_errors_relevant"]
        ),
        "violation_split_involved_fraction": safe_div(
            counts["violation_split_involved"], counts["violations_relevant"]
        ),
        "violation_merge_involved_fraction": safe_div(
            counts["violation_merge_involved"], counts["violations_relevant"]
        ),
        "violation_split_only_fraction": safe_div(
            counts["violation_split_only"], counts["violations_relevant"]
        ),
        "safe_lone_cut_repair_fraction": safe_div(
            counts["safe_lone_cut_repairs"], counts["violations_relevant"]
        ),
        "binary_gradient_direction_accuracy": safe_div(
            counts["binary_gradient_correct_votes"],
            counts["binary_gradient_votes"],
        ),
        "soft_active_cycle_rate_relevant": safe_div(
            counts["soft_active_cycles_relevant"], counts["relevant_cycles"]
        ),
        "soft_gradient_direction_accuracy": safe_div(
            counts["soft_gradient_correct_votes"], counts["soft_gradient_votes"]
        ),
        "mean_soft_cycle_score_relevant": safe_div(
            counts["soft_score_sum_relevant"], counts["relevant_cycles"]
        ),
    }


def make_rectangle_types(long_range: int) -> list[RectangleType]:
    short = [EdgeType(axis, axis, 1) for axis in range(3)]
    long = [EdgeType(axis + 3, axis, long_range) for axis in range(3)]
    rectangles: list[RectangleType] = []
    for axis_a in range(3):
        for axis_b in range(axis_a + 1, 3):
            for first in (short[axis_a], long[axis_a]):
                for second in (short[axis_b], long[axis_b]):
                    rectangles.append(RectangleType(first, second))
    return rectangles


def save_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(overall_rows: Sequence[dict[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    thresholds = np.asarray([float(row["threshold"]) for row in overall_rows])
    violation_rate = np.asarray(
        [float(row["violation_rate_relevant"]) for row in overall_rows]
    )
    base_error = np.asarray(
        [float(row["edge_error_rate_relevant"]) for row in overall_rows]
    )
    violation_error = np.asarray(
        [float(row["edge_error_rate_in_violations"]) for row in overall_rows]
    )
    safe_repair = np.asarray(
        [float(row["safe_lone_cut_repair_fraction"]) for row in overall_rows]
    )
    gradient_accuracy = np.asarray(
        [float(row["binary_gradient_direction_accuracy"]) for row in overall_rows]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(thresholds, violation_rate, marker="o")
    axes[0].set(title="Cycle violations", xlabel="Affinity threshold", ylabel="Rate on relevant cycles")

    axes[1].plot(thresholds, base_error, marker="o", label="All relevant edges")
    axes[1].plot(thresholds, violation_error, marker="o", label="Edges in violations")
    axes[1].set(title="Error enrichment", xlabel="Affinity threshold", ylabel="Affinity error rate")
    axes[1].legend(frameon=False)

    axes[2].plot(thresholds, safe_repair, marker="o", label="Lone-cut push is correct")
    axes[2].plot(thresholds, gradient_accuracy, marker="o", label="4-edge gradient accuracy")
    axes[2].axhline(0.5, color="0.6", linestyle="--", linewidth=1)
    axes[2].set(title="Repair ambiguity", xlabel="Affinity threshold", ylabel="Fraction")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    crops: Sequence[tuple[slice, slice, slice]],
    overall_rows: Sequence[dict[str, object]],
) -> None:
    lines = [
        "# Idea B: offline transitivity audit",
        "",
        "This audit evaluates four-edge cycle consistency in BANIS short- and long-range affinities. "
        "A violation is a rectangular cycle with exactly three predicted join edges. "
        "The decisive ambiguity test checks whether the cycle-loss gradient agrees with GT affinity supervision.",
        "",
        "## Inputs",
        "",
        f"- Prediction: `{args.prediction}`",
        f"- Ground truth: `{args.ground_truth}`",
        f"- Crops: {', '.join(f'`{crop_to_text(c)}`' for c in crops)}",
        f"- Anchor stride: {args.stride}",
        f"- Long-range offset: {args.long_range}",
        "",
        "## Aggregate results",
        "",
        "| threshold | relevant cycles | violations | violation rate | edge error | error in violations | enrichment | safe lone-cut repair | gradient direction accuracy | soft error AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["threshold"], 2),
                    fmt(row["relevant_cycles"], 0),
                    fmt(row["violation_count"], 0),
                    fmt(row["violation_rate_relevant"]),
                    fmt(row["edge_error_rate_relevant"]),
                    fmt(row["edge_error_rate_in_violations"]),
                    fmt(row["edge_error_enrichment"], 2),
                    fmt(row["safe_lone_cut_repair_fraction"]),
                    fmt(row["binary_gradient_direction_accuracy"]),
                    fmt(row["soft_error_auroc"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- Error enrichment > 1 means violations localize unreliable affinity predictions.",
            "- Safe lone-cut repair measures how often increasing the missing fourth edge is correct. "
            "Low values mean the apparent violation is usually caused by false-positive path edges instead.",
            "- Gradient direction accuracy measures all four directions induced by the hinge: increase the target and decrease its three antecedents. "
            "A value near or below 0.5 is not a reliable unlabeled training signal.",
            "- Soft error AUROC measures whether the continuous cycle score ranks cycles containing any thresholded affinity error.",
            "",
            "Detailed per-rectangle results are in `per_rectangle_metrics.csv`; machine-readable aggregate results are in `overall_metrics.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    prediction_zarr = zarr.open(args.prediction, mode="r")
    ground_truth_zarr = zarr.open(args.ground_truth, mode="r")
    if not hasattr(prediction_zarr, "shape") or not hasattr(ground_truth_zarr, "shape"):
        raise TypeError("--prediction and --ground-truth must point directly to Zarr arrays")
    if prediction_zarr.ndim != 4 or prediction_zarr.shape[0] < 6:
        raise ValueError(
            f"Prediction must have shape (>=6, z, y, x), got {prediction_zarr.shape}"
        )
    if ground_truth_zarr.ndim != 3:
        raise ValueError(f"Ground truth must be 3-D, got {ground_truth_zarr.shape}")
    if tuple(prediction_zarr.shape[1:]) != tuple(ground_truth_zarr.shape):
        raise ValueError(
            f"Prediction/GT spatial shapes differ: {prediction_zarr.shape[1:]} vs "
            f"{ground_truth_zarr.shape}"
        )

    if args.crop:
        crops = [parse_crop(text, ground_truth_zarr.shape) for text in args.crop]
    else:
        crops = [tuple(slice(0, int(dim)) for dim in ground_truth_zarr.shape)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rectangles = make_rectangle_types(args.long_range)
    aggregate = {threshold: new_counts() for threshold in args.thresholds}
    aggregate_scores: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in args.thresholds
    }
    aggregate_errors: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in args.thresholds
    }
    rows: list[dict[str, object]] = []

    for crop_index, crop in enumerate(crops):
        print(f"Loading crop {crop_index + 1}/{len(crops)}: {crop_to_text(crop)}", flush=True)
        prediction = np.asarray(
            prediction_zarr[(slice(0, 6),) + crop], dtype=np.float32
        )
        ground_truth = np.asarray(ground_truth_zarr[crop])
        if not np.isfinite(prediction).all():
            raise ValueError(f"Prediction crop {crop_to_text(crop)} contains NaN/Inf")

        for rectangle in rectangles:
            edges, gt_edges, gt_touch = rectangle_edges(
                prediction, ground_truth, rectangle, args.stride
            )
            hinges = soft_cycle_hinges(edges)
            for threshold in args.thresholds:
                counts, scores, errors = compute_counts(
                    edges, gt_edges, gt_touch, threshold, hinges
                )
                add_counts(aggregate[threshold], counts)
                aggregate_scores[threshold].append(scores)
                aggregate_errors[threshold].append(errors)
                row: dict[str, object] = {
                    "crop": crop_to_text(crop),
                    "rectangle": rectangle.name,
                    "scale": rectangle.scale,
                    "threshold": threshold,
                }
                row.update(finalize_counts(counts))
                row["soft_error_auroc"] = binary_auc(scores, errors)
                rows.append(row)

        del prediction, ground_truth

    overall_rows: list[dict[str, object]] = []
    for threshold in args.thresholds:
        row = {"threshold": threshold}
        row.update(finalize_counts(aggregate[threshold]))
        scores = np.concatenate(aggregate_scores[threshold])
        errors = np.concatenate(aggregate_errors[threshold])
        row["soft_error_auroc"] = binary_auc(scores, errors)
        overall_rows.append(row)

    save_csv(rows, output_dir / "per_rectangle_metrics.csv")
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall_rows, indent=2, allow_nan=True) + "\n"
    )
    make_plot(overall_rows, output_dir / "transitivity_audit.png")
    write_markdown_report(
        output_dir / "README.md", args, crops, overall_rows
    )
    print(json.dumps(overall_rows, indent=2, allow_nan=True))
    print(f"Wrote audit to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="Stacked BANIS prediction Zarr array")
    parser.add_argument("--ground-truth", required=True, help="Instance-label Zarr array")
    parser.add_argument(
        "--crop",
        action="append",
        help="Repeatable z0:z1,y0:y1,x0:x1 crop. Defaults to the full volume.",
    )
    parser.add_argument("--stride", type=int, default=4, help="Cycle-anchor sampling stride")
    parser.add_argument("--long-range", type=int, default=10)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8],
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.long_range < 2:
        raise SystemExit("--long-range must be >= 2")
    if any(not (0 < threshold < 1) for threshold in args.thresholds):
        raise SystemExit("All thresholds must lie strictly between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
