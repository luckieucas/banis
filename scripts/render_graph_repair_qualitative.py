#!/usr/bin/env python3
"""Render a reproducible qualitative panel for one graph-repair case."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.lines import Line2D
from skimage.segmentation import find_boundaries

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.sdt_affinity_graph import UnionFind, load_volume  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--edge-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-threshold", type=float, default=0.7)
    parser.add_argument("--maximum-uncertainty", type=float, default=1.0)
    parser.add_argument("--minimum-evidence", type=int, default=8)
    parser.add_argument("--minimum-affinity-mean", type=float, default=0.4)
    parser.add_argument("--minimum-high-fraction", type=float, default=0.5)
    parser.add_argument("--crop-size", type=int, default=768)
    return parser.parse_args()


def read_volume_slice(path: Path, z_index: int) -> np.ndarray:
    if not (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz")):
        array = load_volume(path)
        result = np.asarray(array[z_index])
        if result.ndim == 3 and result.shape[-1] == 1:
            result = result[..., 0]
        return result
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    size = list(reader.GetSize())
    if not (0 <= z_index < size[2]):
        raise ValueError(f"Slice {z_index} outside NIfTI depth {size[2]} for {path}")
    reader.SetExtractIndex([0, 0, z_index])
    reader.SetExtractSize([size[0], size[1], 0])
    return np.asarray(sitk.GetArrayFromImage(reader.Execute()))


def accepted_lut_and_pairs(
    n_labels: int,
    edges: list[dict[str, str]],
    score_by_pair: dict[tuple[int, int], tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    candidates = []
    for row in edges:
        pair = (int(row["fragment_a"]), int(row["fragment_b"]))
        if pair not in score_by_pair:
            raise ValueError(f"Missing learned score for edge {pair}")
        probability, uncertainty = score_by_pair[pair]
        candidates.append((row, probability, uncertainty))
    candidates.sort(
        key=lambda item: (
            -item[1],
            item[2],
            int(item[0]["fragment_a"]),
            int(item[0]["fragment_b"]),
        )
    )
    union_find = UnionFind(n_labels)
    accepted = []
    for row, probability, uncertainty in candidates:
        if int(row["count"]) < args.minimum_evidence:
            continue
        if float(row["mean"]) < args.minimum_affinity_mean:
            continue
        if float(row["high_fraction"]) < args.minimum_high_fraction:
            continue
        if probability < args.probability_threshold or uncertainty > args.maximum_uncertainty:
            continue
        left, right = int(row["fragment_a"]), int(row["fragment_b"])
        if union_find.union(left, right):
            accepted.append((left, right))
    return union_find.compact_lut(), accepted


def label_rgb(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.uint64, copy=False)
    rgb = np.empty((*labels.shape, 3), dtype=np.float32)
    rgb[..., 0] = ((labels * 37 + 17) % 251) / 250.0
    rgb[..., 1] = ((labels * 67 + 43) % 241) / 240.0
    rgb[..., 2] = ((labels * 97 + 71) % 239) / 238.0
    rgb[labels == 0] = 0
    return rgb


def normalized_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    low, high = np.percentile(image, (1, 99))
    return np.clip((image - low) / max(float(high - low), 1e-6), 0, 1)


def crop_bounds(shape: tuple[int, int], center: tuple[float, float], size: int) -> tuple[slice, slice]:
    height, width = shape
    crop_h, crop_w = min(size, height), min(size, width)
    y0 = int(round(center[0] - crop_h / 2))
    x0 = int(round(center[1] - crop_w / 2))
    y0 = min(max(y0, 0), height - crop_h)
    x0 = min(max(x0, 0), width - crop_w)
    return slice(y0, y0 + crop_h), slice(x0, x0 + crop_w)


def main() -> None:
    args = parse_args()
    config = json.loads((args.graph_dir / "config.json").read_text())
    case = config["case"]
    fragments = load_volume(config["fragments"])
    if len(fragments.shape) != 3:
        raise ValueError(f"Expected 3-D fragments, got {fragments.shape}")
    deterministic_lut = np.load(args.graph_dir / "best_affinity_label_lut_analysis_only.npy")
    with (args.graph_dir / "edge_evidence.csv").open(newline="") as handle:
        edges = list(csv.DictReader(handle))
    score_by_pair = {}
    with args.edge_scores.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case"] != case:
                continue
            pair = (int(row["fragment_a"]), int(row["fragment_b"]))
            score_by_pair[pair] = (float(row["probability"]), float(row["uncertainty"]))
    learned_lut, accepted_pairs = accepted_lut_and_pairs(
        len(deterministic_lut), edges, score_by_pair, args
    )

    involved = np.zeros(len(learned_lut), dtype=bool)
    for left, right in accepted_pairs:
        involved[left] = involved[right] = True
    slice_scores = np.zeros(fragments.shape[0], dtype=np.int64)
    for z_index in range(fragments.shape[0]):
        labels = np.asarray(fragments[z_index])
        slice_scores[z_index] = int(np.count_nonzero(involved[labels]))
    z_index = int(np.argmax(slice_scores))
    fragment_slice = np.asarray(fragments[z_index])
    active_pixels = involved[fragment_slice]
    if np.any(active_pixels):
        y_coords, x_coords = np.nonzero(active_pixels)
        center = (float(np.median(y_coords)), float(np.median(x_coords)))
    else:
        center = ((fragment_slice.shape[0] - 1) / 2, (fragment_slice.shape[1] - 1) / 2)
    crop = crop_bounds(fragment_slice.shape, center, args.crop_size)

    image = read_volume_slice(args.image, z_index)[crop]
    ground_truth = read_volume_slice(Path(config["ground_truth"]), z_index)[crop]
    fragment_crop = fragment_slice[crop]
    deterministic_crop = deterministic_lut[fragment_crop]
    learned_crop = learned_lut[fragment_crop]
    image_gray = normalized_image(image)

    # Per-slice fragment centroids support a compact rendering of accepted edges.
    full_labels = fragment_slice
    ys, xs = np.nonzero(full_labels)
    ids = full_labels[ys, xs].astype(np.int64, copy=False)
    counts = np.bincount(ids, minlength=len(learned_lut))
    sum_y = np.bincount(ids, weights=ys, minlength=len(learned_lut))
    sum_x = np.bincount(ids, weights=xs, minlength=len(learned_lut))
    centroids = {}
    present_ids = np.flatnonzero(counts)
    for label in present_ids:
        centroids[int(label)] = (sum_y[label] / counts[label], sum_x[label] / counts[label])

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.7), constrained_layout=True)
    for axis in axes.flat:
        axis.set_axis_off()
    axes[0, 0].imshow(image_gray, cmap="gray")
    axes[0, 0].set_title("Raw EM")
    axes[0, 1].imshow(label_rgb(ground_truth))
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].imshow(label_rgb(fragment_crop))
    axes[0, 2].set_title("SDT fragments")
    axes[1, 0].imshow(label_rgb(deterministic_crop))
    axes[1, 0].set_title("Deterministic graph repair")
    axes[1, 1].imshow(label_rgb(learned_crop))
    axes[1, 1].set_title("Safety-anchored learned gate")
    axes[1, 2].imshow(image_gray, cmap="gray")
    y_offset, x_offset = crop[0].start, crop[1].start
    drawn = 0
    for left, right in accepted_pairs:
        if left not in centroids or right not in centroids:
            continue
        y1, x1 = centroids[left]
        y2, x2 = centroids[right]
        if not (crop[0].start <= y1 < crop[0].stop and crop[1].start <= x1 < crop[1].stop):
            continue
        if not (crop[0].start <= y2 < crop[0].stop and crop[1].start <= x2 < crop[1].stop):
            continue
        axes[1, 2].plot([x1 - x_offset, x2 - x_offset], [y1 - y_offset, y2 - y_offset], color="#ffd43b", lw=1.2)
        axes[1, 2].scatter([x1 - x_offset, x2 - x_offset], [y1 - y_offset, y2 - y_offset], s=8, c="#ff6b6b")
        drawn += 1
    gt_boundary = find_boundaries(ground_truth, mode="inner")
    fragment_boundary = find_boundaries(fragment_crop, mode="inner")
    learned_boundary = find_boundaries(learned_crop, mode="inner")
    overlay = np.zeros((*image_gray.shape, 4), dtype=np.float32)
    overlay[fragment_boundary] = (1.0, 0.15, 0.15, 0.65)
    overlay[learned_boundary] = (0.1, 0.8, 1.0, 0.75)
    overlay[gt_boundary] = (0.2, 1.0, 0.35, 0.95)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f"Accepted edges + boundaries ({drawn} shown)")
    axes[1, 2].legend(
        handles=[
            Line2D([0], [0], color="#ffd43b", lw=2, label="accepted edge"),
            Line2D([0], [0], color="#33ff59", lw=2, label="GT boundary"),
            Line2D([0], [0], color="#ff2626", lw=2, label="SDT boundary"),
            Line2D([0], [0], color="#1accff", lw=2, label="repaired boundary"),
        ],
        loc="lower right",
        fontsize=7,
        framealpha=0.75,
    )
    fig.suptitle(
        f"{case}: z={z_index}, accepted merges={len(accepted_pairs)}, "
        f"p≥{args.probability_threshold:.2f}",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "case": case,
        "z_index": z_index,
        "crop_yx": [[crop[0].start, crop[0].stop], [crop[1].start, crop[1].stop]],
        "accepted_merges": len(accepted_pairs),
        "accepted_edges_rendered_in_slice": drawn,
        "probability_threshold": args.probability_threshold,
        "maximum_uncertainty": args.maximum_uncertainty,
        "minimum_evidence": args.minimum_evidence,
        "minimum_affinity_mean": args.minimum_affinity_mean,
        "minimum_high_fraction": args.minimum_high_fraction,
        "graph_dir": str(args.graph_dir),
        "edge_scores": str(args.edge_scores),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
