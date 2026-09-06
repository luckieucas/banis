#!/usr/bin/env python
"""Sweep affinity threshold and small-component filtering for cerebellum instances."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BANIS import BANIS
from scripts.evaluate_cereb_train_instances import (
    binary_precision_recall_f1,
    instance_metrics_iou,
    iter_sample_dirs,
    summarize,
    to_jsonable,
)
from src.inference.inference import compute_connected_component_segmentation, patched_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one inference pass per sample and sweep instance post-processing settings."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-data-path", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=["cereb_normalized", "cereb_p7_pc2"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--affinity-thresholds",
        nargs="+",
        type=float,
        default=[0.2, 0.3, 0.4, 0.5, 0.6],
    )
    parser.add_argument(
        "--min-pred-voxels",
        nargs="+",
        type=int,
        default=[0, 20, 50, 100, 200],
        help="Remove predicted connected components smaller than each listed voxel count.",
    )
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--small-size", default=None, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--prediction-channels", default=7, type=int, choices=range(3, 8))
    return parser.parse_args()


def filter_small_components(seg: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1:
        return seg
    max_label = int(seg.max())
    if max_label == 0:
        return seg
    sizes = np.bincount(seg.reshape(-1), minlength=max_label + 1)
    keep = sizes >= min_voxels
    keep[0] = False
    filtered = seg.copy()
    filtered[~keep[filtered]] = 0
    return filtered


def metric_key(affinity_threshold: float, min_pred_voxels: int) -> str:
    return f"aff{affinity_threshold:g}_min{min_pred_voxels}"


def evaluate_sample_sweep(
    sample_dir: Path,
    model: BANIS,
    args: argparse.Namespace,
    small_size: int,
) -> List[Dict]:
    data_path = sample_dir / "data.zarr"
    data = zarr.open(str(data_path), mode="r")
    img = data["img"]
    gt = data["seg"][:]
    labeled_mask = gt >= 0
    gt_eval = gt.astype(np.int64, copy=True)
    gt_eval[~labeled_mask] = 0

    dataset = sample_dir.parent.parent.name
    print(f"\n=== Inference {dataset}/{sample_dir.name} ===", flush=True)
    print(f"image={img.shape} {img.dtype}, gt={gt.shape} {gt.dtype}", flush=True)

    pred_aff = patched_inference(
        img,
        model=model,
        small_size=small_size,
        do_overlap=True,
        prediction_channels=args.prediction_channels,
        divide=255,
    )

    sample_results: List[Dict] = []
    for affinity_threshold in args.affinity_thresholds:
        pred_seg = compute_connected_component_segmentation(pred_aff[:3] > affinity_threshold)
        pred_eval_base = pred_seg.astype(np.uint32, copy=False)
        pred_eval_base[~labeled_mask] = 0

        for min_pred_voxels in args.min_pred_voxels:
            pred_eval = filter_small_components(pred_eval_base, min_pred_voxels)
            metrics = instance_metrics_iou(gt_eval, pred_eval, thresh=args.iou_threshold)
            metrics.update(binary_precision_recall_f1(pred_eval > 0, gt_eval > 0))
            metrics.update(
                {
                    "dataset": dataset,
                    "split": sample_dir.parent.name,
                    "sample_id": sample_dir.name,
                    "data_path": str(data_path),
                    "affinity_threshold": affinity_threshold,
                    "min_pred_voxels": min_pred_voxels,
                    "iou_threshold": args.iou_threshold,
                    "setting": metric_key(affinity_threshold, min_pred_voxels),
                }
            )
            sample_results.append({k: to_jsonable(v) for k, v in metrics.items()})
            print(
                f"setting={metrics['setting']} "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
                f"n_pred={metrics['n_pred']}",
                flush=True,
            )
    return sample_results


def summarize_by_setting(results: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for result in results:
        grouped[result["setting"]].append(result)

    summaries = {}
    for setting, setting_results in sorted(grouped.items()):
        summary = summarize(setting_results)
        first = setting_results[0]
        summary.update(
            {
                "affinity_threshold": first["affinity_threshold"],
                "min_pred_voxels": first["min_pred_voxels"],
            }
        )
        summaries[setting] = summary
    return summaries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = iter_sample_dirs(args)
    print(f"Selected {len(sample_dirs)} samples", flush=True)
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)

    model = BANIS.load_from_checkpoint(str(args.checkpoint))
    model.eval()
    model.cuda()
    torch.set_float32_matmul_precision("medium")
    small_size = args.small_size or int(model.hparams.small_size)

    results: List[Dict] = []
    for sample_dir in sample_dirs:
        results.extend(evaluate_sample_sweep(sample_dir, model, args, small_size))
        partial = {
            "checkpoint": str(args.checkpoint),
            "base_data_path": str(args.base_data_path),
            "datasets": args.datasets,
            "split": args.split,
            "iou_threshold": args.iou_threshold,
            "affinity_thresholds": args.affinity_thresholds,
            "min_pred_voxels": args.min_pred_voxels,
            "results": results,
            "summary_by_setting": summarize_by_setting(results),
        }
        with (args.output_dir / "partial_sweep_results.json").open("w") as f:
            json.dump(partial, f, indent=2)

    output = {
        "checkpoint": str(args.checkpoint),
        "base_data_path": str(args.base_data_path),
        "datasets": args.datasets,
        "split": args.split,
        "iou_threshold": args.iou_threshold,
        "affinity_thresholds": args.affinity_thresholds,
        "min_pred_voxels": args.min_pred_voxels,
        "results": results,
        "summary_by_setting": summarize_by_setting(results),
    }
    output_path = args.output_dir / "sweep_instance_metrics.json"
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Best settings by overall_f1 ===", flush=True)
    ranked = sorted(
        output["summary_by_setting"].items(),
        key=lambda item: item[1]["overall_f1"],
        reverse=True,
    )
    for setting, summary in ranked[:10]:
        print(
            f"{setting}: overall_f1={summary['overall_f1']:.4f}, "
            f"overall_acc={summary['overall_accuracy']:.4f}, "
            f"precision={summary['overall_precision']:.4f}, "
            f"recall={summary['overall_recall']:.4f}, "
            f"pred={summary['total_pred_instances']}, true={summary['total_true_instances']}",
            flush=True,
        )
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
