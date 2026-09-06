#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <dataset-slug> <dataset-dir> <validation-case> <spacing-nm-zyx-comma-separated>" >&2
  exit 2
fi

DATASET_SLUG="$1"
DATASET_DIR="$2"
CASE="$3"
IFS=',' read -r SPACING_Z SPACING_Y SPACING_X <<<"$4"

REPO="/projects/weilab/liupeng/code/projects/banis"
TRAIN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train"
CKPT="${TRAIN_ROOT}/checkpoints/mitoem2_all_sdt_mednextB_lr3e-4_s0_b2_k3_ns250000/default/checkpoints/epoch=171-step=250000.ckpt"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation/${DATASET_SLUG}/${CASE}"
PRED_DIR="${RUN_ROOT}/predictions"
DECODE_DIR="${RUN_ROOT}/sdt_fragments"
GRAPH_DIR="${RUN_ROOT}/graph_audit"
IMAGE="${DATASET_DIR}/imagesTr/${CASE}_0000.nii.gz"
GROUND_TRUTH="${DATASET_DIR}/labelsTr/${CASE}.nii.gz"
MASK="${DATASET_DIR}/masksTr/${CASE}_mask.nii.gz"

for REQUIRED in "${CKPT}" "${IMAGE}" "${GROUND_TRUTH}"; do
  if [[ ! -f "${REQUIRED}" ]]; then
    echo "Missing required input: ${REQUIRED}" >&2
    exit 1
  fi
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO}"
mkdir -p "${PRED_DIR}" "${DECODE_DIR}" "${GRAPH_DIR}"

ALL_CHANNELS_PRESENT=true
for CHANNEL in 0 1 2 3 4 5 6; do
  if [[ ! -d "${PRED_DIR}/channel_${CHANNEL}.zarr" ]]; then
    ALL_CHANNELS_PRESENT=false
  fi
done
if [[ "${ALL_CHANNELS_PRESENT}" != true ]]; then
  echo "=== ${CASE}: full seven-channel prediction ==="
  micromamba run -n sdt python src/inference/pred_and_eval.py \
    --checkpoint_path "${CKPT}" \
    --input_path "${IMAGE}" \
    --output_path "${PRED_DIR}" \
    --prediction_channels 7 \
    --small_size 128 \
    --use_batch \
    --batch_size 2 \
    --save_channels 0,1,2,3,4,5,6 \
    --output_format zarr \
    --padding_mode edge \
    --padding_position center
else
  echo "${CASE}: all prediction channels exist; skipping inference"
fi

if [[ ! -f "${DECODE_DIR}/final_instance_seg.tiff" ]]; then
  echo "=== ${CASE}: SDT fragment generation ==="
  micromamba run -n sdt python src/inference/watershed_and_eval_sdt_only.py \
    --prediction_file "${PRED_DIR}/channel_6.zarr" \
    --output_path "${DECODE_DIR}" \
    --skeleton_threshold 0.5 \
    --foreground_threshold 0.0 \
    --output_format tiff \
    --edt_parallel "${SLURM_CPUS_PER_TASK:-8}" \
    --edt_downsample_factor 2
else
  echo "${CASE}: fragments exist; skipping SDT decode"
fi

MASK_ARGS=()
if [[ -f "${MASK}" ]]; then
  MASK_ARGS=(--mask "${MASK}")
fi

if [[ ! -f "${GRAPH_DIR}/SUCCESS" ]]; then
  echo "=== ${CASE}: six-channel held-out graph audit ==="
  micromamba run -n sdt python scripts/evaluate_sdt_affinity_graph.py \
    --case "${CASE}" \
    --fragments "${DECODE_DIR}/final_instance_seg.tiff" \
    --ground-truth "${GROUND_TRUTH}" \
    "${MASK_ARGS[@]}" \
    --affinity "0=${PRED_DIR}/channel_0.zarr" \
    --affinity "1=${PRED_DIR}/channel_1.zarr" \
    --affinity "2=${PRED_DIR}/channel_2.zarr" \
    --affinity "3=${PRED_DIR}/channel_3.zarr" \
    --affinity "4=${PRED_DIR}/channel_4.zarr" \
    --affinity "5=${PRED_DIR}/channel_5.zarr" \
    --thresholds 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
    --minimum-evidence 8 \
    --minimum-high-fraction 0.50 \
    --oracle-minimum-purity 0.90 \
    --affinity-positive-threshold 0.50 \
    --block-shape 256 512 512 \
    --voxel-spacing-nm "${SPACING_Z}" "${SPACING_Y}" "${SPACING_X}" \
    --output-dir "${GRAPH_DIR}"
else
  echo "${CASE}: graph audit SUCCESS exists; skipping"
fi

echo "Completed held-out validation ${DATASET_SLUG}/${CASE}: ${RUN_ROOT}"
