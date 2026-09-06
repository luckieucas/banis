#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
CHECKPOINT="/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train/checkpoints/mitoem2_all_sdt_mednextB_lr3e-4_s0_b2_k3_ns250000/default/checkpoints/epoch=171-step=250000.ckpt"
INPUT="/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra/imagesTs/me2-pyra_test01_0000.nii.gz"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/streaming_inference/pyra_test01_seed0"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

JOB=$(sbatch --parsable --partition long --gres gpu:l40s:1 \
  --cpus-per-task 8 --mem 128G --time 1-00:00:00 \
  --job-name banis_stream_pyra \
  --output "${LOG_ROOT}/predict.%j.out" \
  --error "${LOG_ROOT}/predict.%j.err" \
  --wrap "cd '${REPO}' && PYTHONPATH='${REPO}:${REPO}/src' '${PYTHON}' scripts/stream_predict_affinities.py --checkpoint-path '${CHECKPOINT}' --input-path '${INPUT}' --output-dir '${OUTPUT_ROOT}/prediction' --save-channels 0 1 2 3 4 5 6 --patch-size 128 --batch-size 2 --output-block-shape 256 512 512 --output-chunks 64 256 256 --padding-mode edge --padding-position center --seed 0")

echo "${JOB%%;*}"
