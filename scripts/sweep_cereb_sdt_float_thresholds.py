#!/usr/bin/env python
import argparse
import json
import sys
import time
from pathlib import Path

import cc3d
import edt
import numpy as np
import tifffile
import torch
import zarr
from scipy.ndimage import zoom
from skimage.segmentation import watershed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BANIS import BANIS
from src.inference.inference import patched_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BANIS float SDT inference and sweep SDT-only watershed seed thresholds."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-channels", default=7, type=int)
    parser.add_argument("--small-size", default=128, type=int)
    parser.add_argument("--thresholds", nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=float)
    parser.add_argument("--min-size", default=200, type=int)
    parser.add_argument("--edt-downsample-factor", default=2, type=int)
    parser.add_argument("--edt-parallel", default=8, type=int)
    parser.add_argument("--save-final-tiffs", action="store_true")
    parser.add_argument("--reuse-prediction", action="store_true")
    return parser.parse_args()


def sample_key(sample_dir: Path) -> str:
    return f"{sample_dir.parent.parent.name}__{sample_dir.name}"


def instance_metrics_iou_masked(y_true, y_pred, valid, thresh=0.5):
    true_pos = (y_true > 0) & valid
    pred_pos = (y_pred > 0) & valid
    true_vals = y_true[true_pos]
    pred_vals = y_pred[pred_pos]
    true_ids, true_inv = np.unique(true_vals, return_inverse=True)
    pred_ids, pred_inv = np.unique(pred_vals, return_inverse=True)
    n_true = int(true_ids.size)
    n_pred = int(pred_ids.size)
    if n_true == 0 and n_pred == 0:
        return dict(tp=0, fp=0, fn=0, precision=0.0, recall=0.0, accuracy=0.0, f1=0.0, n_true=0, n_pred=0)

    true_area = np.bincount(true_inv + 1, minlength=n_true + 1) if n_true else np.zeros(1, dtype=np.int64)
    pred_area = np.bincount(pred_inv + 1, minlength=n_pred + 1) if n_pred else np.zeros(1, dtype=np.int64)
    del true_vals, pred_vals, true_inv, pred_inv

    overlap = true_pos & pred_pos
    if np.any(overlap) and n_true and n_pred:
        ov_true = y_true[overlap]
        ov_pred = y_pred[overlap]
        true_seq = np.searchsorted(true_ids, ov_true).astype(np.int64, copy=False) + 1
        pred_seq = np.searchsorted(pred_ids, ov_pred).astype(np.int64, copy=False) + 1
        pair_index = true_seq * (n_pred + 1) + pred_seq
        overlap_counts = np.bincount(pair_index)
        nonzero = np.nonzero(overlap_counts)[0]
        true_pair = nonzero // (n_pred + 1)
        pred_pair = nonzero % (n_pred + 1)
        keep = (true_pair > 0) & (pred_pair > 0)
        true_pair = true_pair[keep]
        pred_pair = pred_pair[keep]
        inter = overlap_counts[nonzero[keep]].astype(np.float64)
        union = true_area[true_pair] + pred_area[pred_pair] - inter
        ious = inter / union
    else:
        ious = np.empty(0, dtype=np.float64)

    tp = int(np.sum(ious >= thresh))
    fp = int(n_pred - tp)
    fn = int(n_true - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, accuracy=accuracy, f1=f1, n_true=n_true, n_pred=n_pred)


def binary_metrics_masked(y_pred, y_true, valid):
    pred = (y_pred > 0) & valid
    true = (y_true > 0) & valid
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true & valid))
    fn = int(np.sum(~pred & true & valid))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(binary_tp=tp, binary_fp=fp, binary_fn=fn, binary_precision=precision, binary_recall=recall, binary_f1=f1, binary_accuracy=accuracy)


def compute_distance(foreground, downsample_factor, parallel):
    if downsample_factor <= 1:
        return edt.edt(
            foreground.astype(np.uint8),
            anisotropy=tuple([1.0] * foreground.ndim),
            black_border=True,
            parallel=parallel,
        )
    original_shape = foreground.shape
    foreground_ds = zoom(foreground.astype(np.float32), 1.0 / downsample_factor, order=0) > 0.5
    distance_ds = edt.edt(
        foreground_ds.astype(np.uint8),
        anisotropy=tuple([1.0] * foreground_ds.ndim),
        black_border=True,
        parallel=parallel,
    )
    upsample_zoom = [orig / ds for orig, ds in zip(original_shape, distance_ds.shape)]
    return zoom(distance_ds, upsample_zoom, order=1) * downsample_factor


def remove_small_components(seg, min_size):
    unique_labels, counts = np.unique(seg, return_counts=True)
    remove_labels = unique_labels[(unique_labels > 0) & (counts < min_size)]
    if remove_labels.size:
        remove = np.zeros(int(unique_labels.max()) + 1, dtype=bool)
        remove[remove_labels.astype(np.int64)] = True
        seg[remove[seg]] = 0
    return int(remove_labels.size)


def summarize(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(total_tp=tp, total_fp=fp, total_fn=fn, overall_precision=precision, overall_recall=recall, overall_f1=f1, overall_accuracy=accuracy)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = sample_key(args.sample_dir)
    pred_dir = args.output_dir / "predictions_float_zarr"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{key}.zarr"

    data_path = args.sample_dir / "data.zarr"
    data = zarr.open(str(data_path), mode="r")
    img = data["img"]
    gt = data["seg"][:]
    valid = gt >= 0

    started = time.time()
    if pred_path.exists() and args.reuse_prediction:
        print(f"Reusing raw prediction: {pred_path}", flush=True)
        pred = zarr.open(str(pred_path), mode="r")
        sdt = pred[args.prediction_channels - 1].astype(np.float32)
    else:
        print(f"Loading checkpoint: {args.checkpoint}", flush=True)
        model = BANIS.load_from_checkpoint(str(args.checkpoint))
        model.eval()
        model.cuda()
        torch.set_float32_matmul_precision("medium")
        print(f"Running inference for {key}: img={img.shape} dtype={img.dtype}", flush=True)
        pred_aff = patched_inference(
            img,
            model=model,
            small_size=args.small_size,
            do_overlap=True,
            prediction_channels=args.prediction_channels,
            divide=255,
        )
        print(f"Prediction ready: shape={pred_aff.shape} dtype={pred_aff.dtype}", flush=True)
        zarr.array(
            pred_aff.astype(np.float16, copy=False),
            dtype=np.float16,
            store=str(pred_path),
            chunks=(1, 64, 256, 256),
            overwrite=True,
        )
        sdt = pred_aff[args.prediction_channels - 1].astype(np.float32, copy=True)
        del pred_aff, model
        torch.cuda.empty_cache()
        print(f"Saved raw prediction: {pred_path}", flush=True)

    finite = np.isfinite(sdt)
    stats = {
        "min": float(sdt[finite].min()) if np.any(finite) else 0.0,
        "max": float(sdt[finite].max()) if np.any(finite) else 0.0,
        "mean": float(sdt[finite].mean()) if np.any(finite) else 0.0,
        "quantiles": [float(x) for x in np.percentile(sdt[finite], [1, 5, 10, 25, 50, 75, 90, 95, 99])] if np.any(finite) else [],
    }
    print(f"SDT stats: {stats}", flush=True)

    foreground = sdt > 0
    print(f"Foreground voxels: {int(foreground.sum())} / {foreground.size}", flush=True)
    distance = compute_distance(foreground, args.edt_downsample_factor, args.edt_parallel)
    print("Distance transform ready", flush=True)

    tiff_dir = args.output_dir / "sdt_float_threshold_tiffs" / key
    if args.save_final_tiffs:
        tiff_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for threshold in args.thresholds:
        t0 = time.time()
        seeds = cc3d.connected_components(sdt > threshold, connectivity=26)
        n_seed = int(seeds.max()) if seeds.size else 0
        if n_seed == 0:
            final = np.zeros_like(seeds, dtype=np.uint32)
        else:
            final = watershed(-distance, seeds, mask=foreground)
        removed = remove_small_components(final, args.min_size)
        final[~valid] = 0

        metrics = instance_metrics_iou_masked(gt, final, valid, 0.5)
        metrics.update(binary_metrics_masked(final, gt, valid))
        metrics.update(
            {
                "dataset": args.sample_dir.parent.parent.name,
                "sample_id": args.sample_dir.name,
                "sample_key": key,
                "threshold": threshold,
                "n_seed": n_seed,
                "removed_small_components": removed,
                "raw_prediction_path": str(pred_path),
                "data_path": str(data_path),
                "elapsed_sec": time.time() - t0,
            }
        )
        if args.save_final_tiffs:
            tiff_path = tiff_dir / f"threshold_{threshold:g}_final_instance_seg.tif"
            tifffile.imwrite(tiff_path, final.astype(np.uint32, copy=False), bigtiff=True, compression="zlib")
            metrics["final_tiff_path"] = str(tiff_path)
        rows.append(metrics)
        print(
            f"thr={threshold:g}: f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
            f"tp/fp/fn={metrics['tp']}/{metrics['fp']}/{metrics['fn']} "
            f"true/pred={metrics['n_true']}/{metrics['n_pred']} seeds={n_seed} removed={removed} "
            f"time={metrics['elapsed_sec']:.1f}s",
            flush=True,
        )

    output = {
        "sample_key": key,
        "checkpoint": str(args.checkpoint),
        "data_path": str(data_path),
        "raw_prediction_path": str(pred_path),
        "prediction_channel": args.prediction_channels - 1,
        "thresholds": args.thresholds,
        "min_size": args.min_size,
        "edt_downsample_factor": args.edt_downsample_factor,
        "edt_parallel": args.edt_parallel,
        "iou_threshold": 0.5,
        "valid_region": "gt >= 0",
        "sdt_stats": stats,
        "results": rows,
        "summary": summarize(rows),
        "elapsed_sec": time.time() - started,
    }
    out_path = args.output_dir / f"{key}_sdt_float_threshold_metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved metrics: {out_path}", flush=True)


if __name__ == "__main__":
    main()
