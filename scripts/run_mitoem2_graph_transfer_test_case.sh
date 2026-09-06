#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 <dataset-slug> <dataset-dir> <test-case> <spacing-nm-zyx-comma-separated> <fixed-validation-threshold>" >&2
  exit 2
fi

DATASET_SLUG="$1"
DATASET_DIR="$2"
CASE="$3"
IFS=',' read -r SPACING_Z SPACING_Y SPACING_X <<<"$4"
FIXED_THRESHOLD="$5"

REPO="/projects/weilab/liupeng/code/projects/banis"
TRAIN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train"
CKPT="${TRAIN_ROOT}/checkpoints/mitoem2_all_sdt_mednextB_lr3e-4_s0_b2_k3_ns250000/default/checkpoints/epoch=171-step=250000.ckpt"
CACHED_EVAL_ROOT="${TRAIN_ROOT}/universal_generalist_mednextB_test_eval/${DATASET_SLUG}"
FRAGMENTS="${CACHED_EVAL_ROOT}/sdt_only_decode_tiff/universal_mednextB_step250000_sdt_thr0p5/${CASE}/final_instance_seg.tiff"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test/${DATASET_SLUG}/${CASE}"
PRED_DIR="${RUN_ROOT}/predictions"
GRAPH_DIR="${RUN_ROOT}/graph_eval_threshold_${FIXED_THRESHOLD}"
IMAGE="${DATASET_DIR}/imagesTs/${CASE}_0000.nii.gz"
GROUND_TRUTH="${DATASET_DIR}/labelsTs/${CASE}.nii.gz"
MASK="${DATASET_DIR}/masksTs/${CASE}_mask.nii.gz"

for REQUIRED in "${CKPT}" "${FRAGMENTS}" "${IMAGE}" "${GROUND_TRUTH}"; do
  if [[ ! -f "${REQUIRED}" ]]; then
    echo "Missing required input: ${REQUIRED}" >&2
    exit 1
  fi
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO}"
mkdir -p "${PRED_DIR}" "${GRAPH_DIR}"

ALL_AFFINITIES_PRESENT=true
for CHANNEL in 0 1 2 3 4 5; do
  if [[ ! -d "${PRED_DIR}/channel_${CHANNEL}.zarr" ]]; then
    ALL_AFFINITIES_PRESENT=false
  fi
done
if [[ "${ALL_AFFINITIES_PRESENT}" != true ]]; then
  echo "=== ${CASE}: full affinity prediction for fixed-threshold test ==="
  micromamba run -n sdt python src/inference/pred_and_eval.py \
    --checkpoint_path "${CKPT}" \
    --input_path "${IMAGE}" \
    --output_path "${PRED_DIR}" \
    --prediction_channels 7 \
    --small_size 128 \
    --use_batch \
    --batch_size 2 \
    --save_channels 0,1,2,3,4,5 \
    --output_format zarr \
    --padding_mode edge \
    --padding_position center
fi

MASK_ARGS=()
if [[ -f "${MASK}" ]]; then
  MASK_ARGS=(--mask "${MASK}")
fi

if [[ ! -f "${GRAPH_DIR}/SUCCESS" ]]; then
  echo "=== ${CASE}: test evaluation at validation-selected threshold ${FIXED_THRESHOLD} ==="
  micromamba run -n sdt python scripts/evaluate_sdt_affinity_graph.py \
    --case "${CASE}" \
    --fragments "${FRAGMENTS}" \
    --ground-truth "${GROUND_TRUTH}" \
    "${MASK_ARGS[@]}" \
    --affinity "0=${PRED_DIR}/channel_0.zarr" \
    --affinity "1=${PRED_DIR}/channel_1.zarr" \
    --affinity "2=${PRED_DIR}/channel_2.zarr" \
    --affinity "3=${PRED_DIR}/channel_3.zarr" \
    --affinity "4=${PRED_DIR}/channel_4.zarr" \
    --affinity "5=${PRED_DIR}/channel_5.zarr" \
    --thresholds "${FIXED_THRESHOLD}" \
    --minimum-evidence 8 \
    --minimum-high-fraction 0.50 \
    --oracle-minimum-purity 0.90 \
    --affinity-positive-threshold 0.50 \
    --block-shape 256 512 512 \
    --voxel-spacing-nm "${SPACING_Z}" "${SPACING_Y}" "${SPACING_X}" \
    --output-dir "${GRAPH_DIR}"
fi

echo "Completed fixed-threshold test ${DATASET_SLUG}/${CASE}: ${RUN_ROOT}"
