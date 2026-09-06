#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
CHECKPOINT="/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train/checkpoints/mitoem2_all_sdt_mednextB_lr3e-4_s0_b2_k3_ns250000/default/checkpoints/epoch=171-step=250000.ckpt"
INPUT="/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset008_ME2-Stem/imagesTs/me2-stem_test01_0000.nii.gz"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/streaming_inference/real_crop_equivalence"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

JOB=$(sbatch --parsable --partition short --nodelist g013 \
  --gres gpu:l40s:1 --cpus-per-task 4 --mem 64G --time 04:00:00 \
  --job-name banis_stream_equiv \
  --output "${LOG_ROOT}/validate.%j.out" \
  --error "${LOG_ROOT}/validate.%j.err" \
  --wrap "cd '${REPO}' && PYTHONPATH='${REPO}/src' '${PYTHON}' scripts/validate_streaming_inference.py --checkpoint '${CHECKPOINT}' --input '${INPUT}' --output-dir '${OUTPUT_ROOT}' --crop-shape 192 192 192 --patch-size 128 --batch-size 2")

echo "${JOB%%;*}"
