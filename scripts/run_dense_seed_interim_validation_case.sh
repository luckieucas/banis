#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 <seed> <checkpoint> <dataset-slug> <dataset-dir> <validation-case> <spacing-nm-zyx> <output-root>" >&2
  exit 2
fi

SEED="$1"
CKPT="$2"
DATASET_SLUG="$3"
DATASET_DIR="$4"
CASE="$5"
IFS=',' read -r SPACING_Z SPACING_Y SPACING_X <<<"$6"
OUTPUT_ROOT="$7"

if [[ ! "${SEED}" =~ ^[12]$ ]]; then
  echo "Seed must be 1 or 2, got ${SEED}" >&2
  exit 2
fi

REPO="/projects/weilab/liupeng/code/projects/banis"
RUN_ROOT="${OUTPUT_ROOT}/seed${SEED}/${DATASET_SLUG}/${CASE}"
PRED_DIR="${RUN_ROOT}/predictions"
DECODE_DIR="${RUN_ROOT}/sdt_fragments"
GRAPH_DIR="${RUN_ROOT}/graph_eval_threshold_0.40"
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

echo "seed=${SEED}"
echo "checkpoint=${CKPT}"
sha256sum "${CKPT}"

ALL_CHANNELS_PRESENT=true
for CHANNEL in 0 1 2 3 4 5 6; do
  if [[ ! -d "${PRED_DIR}/channel_${CHANNEL}.zarr" ]]; then
    ALL_CHANNELS_PRESENT=false
  fi
done
if [[ "${ALL_CHANNELS_PRESENT}" != true ]]; then
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
fi

if [[ ! -f "${DECODE_DIR}/final_instance_seg.tiff" ]]; then
  micromamba run -n sdt python src/inference/watershed_and_eval_sdt_only.py \
    --prediction_file "${PRED_DIR}/channel_6.zarr" \
    --output_path "${DECODE_DIR}" \
    --skeleton_threshold 0.5 \
    --foreground_threshold 0.0 \
    --output_format tiff \
    --edt_parallel "${SLURM_CPUS_PER_TASK:-8}" \
    --edt_downsample_factor 2
fi

MASK_ARGS=()
if [[ -f "${MASK}" ]]; then
  MASK_ARGS=(--mask "${MASK}")
fi

if [[ ! -f "${GRAPH_DIR}/SUCCESS" ]]; then
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
    --thresholds 0.40 \
    --minimum-evidence 8 \
    --minimum-high-fraction 0.50 \
    --oracle-minimum-purity 0.90 \
    --affinity-positive-threshold 0.50 \
    --block-shape 256 512 512 \
    --voxel-spacing-nm "${SPACING_Z}" "${SPACING_Y}" "${SPACING_X}" \
    --output-dir "${GRAPH_DIR}"
fi

echo "Completed interim validation ${DATASET_SLUG}/${CASE} for seed ${SEED}"
