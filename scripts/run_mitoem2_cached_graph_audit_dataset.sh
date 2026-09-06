#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <dataset-slug> <dataset-dir> <spacing-nm-zyx-comma-separated>" >&2
  exit 2
fi

DATASET_SLUG="$1"
DATASET_DIR="$2"
IFS=',' read -r SPACING_Z SPACING_Y SPACING_X <<<"$3"

REPO="/projects/weilab/liupeng/code/projects/banis"
EVAL_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train/universal_generalist_mednextB_test_eval"
CKPT_TAG="universal_mednextB_step250000"
PRED_ROOT="${EVAL_ROOT}/${DATASET_SLUG}/predictions/${CKPT_TAG}"
DECODE_ROOT="${EVAL_ROOT}/${DATASET_SLUG}/sdt_only_decode_tiff/${CKPT_TAG}_sdt_thr0p5"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/cached_test_analysis/${DATASET_SLUG}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO}"
mkdir -p "${OUTPUT_ROOT}"

mapfile -t CASES < <(
  find "${DECODE_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)
if [[ "${#CASES[@]}" -eq 0 ]]; then
  echo "No decoded cases found under ${DECODE_ROOT}" >&2
  exit 1
fi

for CASE in "${CASES[@]}"; do
  FRAGMENTS="${DECODE_ROOT}/${CASE}/final_instance_seg.tiff"
  GROUND_TRUTH="${DATASET_DIR}/labelsTs/${CASE}.nii.gz"
  AFFINITY="${PRED_ROOT}/${CASE}/channel_4.zarr"
  MASK="${DATASET_DIR}/masksTs/${CASE}_mask.nii.gz"
  OUTPUT_DIR="${OUTPUT_ROOT}/${CASE}"

  for REQUIRED in "${FRAGMENTS}" "${GROUND_TRUTH}" "${AFFINITY}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
      echo "Missing required input: ${REQUIRED}" >&2
      exit 1
    fi
  done
  if [[ -f "${OUTPUT_DIR}/SUCCESS" ]]; then
    echo "${CASE}: SUCCESS exists; skipping"
    continue
  fi

  MASK_ARGS=()
  if [[ -f "${MASK}" ]]; then
    MASK_ARGS=(--mask "${MASK}")
  fi

  echo "=== ${CASE}: cached-test graph audit ==="
  micromamba run -n sdt python scripts/evaluate_sdt_affinity_graph.py \
    --case "${CASE}" \
    --fragments "${FRAGMENTS}" \
    --ground-truth "${GROUND_TRUTH}" \
    "${MASK_ARGS[@]}" \
    --affinity "4=${AFFINITY}" \
    --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
    --minimum-evidence 8 \
    --minimum-high-fraction 0.50 \
    --oracle-minimum-purity 0.90 \
    --affinity-positive-threshold 0.50 \
    --block-shape 32 256 256 \
    --voxel-spacing-nm "${SPACING_Z}" "${SPACING_Y}" "${SPACING_X}" \
    --output-dir "${OUTPUT_DIR}"
done

echo "Completed ${DATASET_SLUG}: ${OUTPUT_ROOT}"
