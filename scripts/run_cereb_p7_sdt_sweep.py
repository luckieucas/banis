#!/usr/bin/env python
"""Run SDT-only P7 evaluation for one BANIS checkpoint and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prediction-channels", default=7, type=int)
    parser.add_argument("--small-size", default=192, type=int)
    parser.add_argument("--thresholds", nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], type=float)
    parser.add_argument("--min-size", default=2500, type=int)
    parser.add_argument("--edt-parallel", default=8, type=int)
    parser.add_argument("--edt-downsample-factor", default=2, type=int)
    parser.add_argument("--save-final-tiffs", action="store_true")
    parser.add_argument("--reuse-prediction", action="store_true")
    parser.add_argument("--label", default="", help="Short model label for reports.")
    return parser.parse_args()


def summarize(group_rows):
    tp = sum(int(r["tp"]) for r in group_rows)
    fp = sum(int(r["fp"]) for r in group_rows)
    fn = sum(int(r["fn"]) for r in group_rows)
    btp = sum(int(r["binary_tp"]) for r in group_rows)
    bfp = sum(int(r["binary_fp"]) for r in group_rows)
    bfn = sum(int(r["binary_fn"]) for r in group_rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    bprecision = btp / (btp + bfp) if btp + bfp else 0.0
    brecall = btp / (btp + bfn) if btp + bfn else 0.0
    bf1 = 2 * bprecision * brecall / (bprecision + brecall) if bprecision + brecall else 0.0
    bacc = btp / (btp + bfp + bfn) if btp + bfp + bfn else 0.0
    return {
        "n_samples": len({r["sample_id"] for r in group_rows}),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "binary_precision": bprecision,
        "binary_recall": brecall,
        "binary_f1": bf1,
        "binary_accuracy": bacc,
    }


def run_sample(args, sample_dir: Path):
    command = [
        sys.executable,
        "scripts/sweep_cereb_sdt_float_thresholds.py",
        "--checkpoint",
        str(args.checkpoint),
        "--sample-dir",
        str(sample_dir),
        "--output-dir",
        str(args.output_dir),
        "--prediction-channels",
        str(args.prediction_channels),
        "--small-size",
        str(args.small_size),
        "--thresholds",
        *[str(x) for x in args.thresholds],
        "--min-size",
        str(args.min_size),
        "--edt-parallel",
        str(args.edt_parallel),
        "--edt-downsample-factor",
        str(args.edt_downsample_factor),
    ]
    if args.save_final_tiffs:
        command.append("--save-final-tiffs")
    if args.reuse_prediction:
        command.append("--reuse-prediction")
    subprocess.run(command, check=True)


def aggregate(args):
    metric_files = sorted(args.output_dir.glob("*_sdt_float_threshold_metrics.json"))
    rows = []
    for path in metric_files:
        data = json.loads(path.read_text())
        for row in data["results"]:
            row = dict(row)
            row["metrics_file"] = str(path)
            rows.append(row)

    summary = []
    for threshold in sorted({float(r["threshold"]) for r in rows}):
        group_rows = [r for r in rows if float(r["threshold"]) == threshold]
        rec = summarize(group_rows)
        rec["threshold"] = threshold
        summary.append(rec)

    per_sample_csv = args.output_dir / "threshold_sweep_per_sample.csv"
    with per_sample_csv.open("w", newline="") as f:
        fieldnames = [
            "sample_id",
            "threshold",
            "f1",
            "accuracy",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "n_true",
            "n_pred",
            "binary_f1",
            "binary_accuracy",
            "binary_precision",
            "binary_recall",
            "n_seed",
            "removed_small_components",
            "final_tiff_path",
            "raw_prediction_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x["sample_id"], float(x["threshold"]))):
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary_csv = args.output_dir / "threshold_sweep_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fieldnames = [
            "threshold",
            "n_samples",
            "f1",
            "accuracy",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "binary_f1",
            "binary_accuracy",
            "binary_precision",
            "binary_recall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in fieldnames})

    best = max(summary, key=lambda r: r["f1"]) if summary else None
    report = [
        f"# P7 SDT-Only Evaluation: {args.label or args.checkpoint.parent.parent.parent.name}",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Test data: `{args.data_root / 'cereb_p7_pc2' / 'test'}`",
        f"- Decode: SDT-only 3D watershed from float SDT channel",
        f"- Thresholds: `{' '.join(str(x) for x in args.thresholds)}`",
        f"- Min component size: `{args.min_size}` voxels",
        "- IoU threshold: `0.5`",
        "",
    ]
    if best:
        report.extend(
            [
                f"- Best threshold: `{best['threshold']}`",
                f"- Best instance F1: `{best['f1']:.4f}`",
                f"- Accuracy@0.5: `{best['accuracy']:.4f}`",
                f"- Precision/Recall: `{best['precision']:.4f}` / `{best['recall']:.4f}`",
                "",
            ]
        )
    report.extend(
        [
            "| Threshold | F1 | Acc@0.5 | Precision | Recall | TP | FP | FN | Binary F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        report.append(
            f"| {row['threshold']:.3f} | {row['f1']:.4f} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['binary_f1']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{summary_csv}`",
            f"- `{per_sample_csv}`",
            f"- `{args.output_dir / 'predictions_float_zarr'}`",
            f"- `{args.output_dir / 'sdt_float_threshold_tiffs'}`",
        ]
    )
    report_path = args.output_dir / "threshold_sweep_report.md"
    report_path.write_text("\n".join(report) + "\n")

    metrics_path = args.output_dir / "threshold_sweep_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "checkpoint": str(args.checkpoint),
                "data_root": str(args.data_root),
                "output_dir": str(args.output_dir),
                "summary": summary,
                "best": best,
                "n_metric_files": len(metric_files),
            },
            indent=2,
        )
    )
    print(f"Wrote {report_path}")
    if best:
        print(f"Best: threshold={best['threshold']} f1={best['f1']:.4f} acc={best['accuracy']:.4f}")


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    test_root = args.data_root / "cereb_p7_pc2" / "test"
    sample_dirs = sorted(p for p in test_root.iterdir() if p.is_dir())
    if not sample_dirs:
        raise RuntimeError(f"No P7 test samples found under {test_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(sample_dirs)} samples from {test_root}")
    for sample_dir in sample_dirs:
        print(f"Evaluating {sample_dir.name}", flush=True)
        run_sample(args, sample_dir)
    aggregate(args)


if __name__ == "__main__":
    main()
