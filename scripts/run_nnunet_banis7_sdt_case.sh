#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 CASE INPUT_NIFTI GROUND_TRUTH MASK_OR_DASH OUTPUT_DIR" >&2
  exit 2
fi

CASE="$1"
INPUT="$2"
GROUND_TRUTH="$3"
MASK="$4"
OUTPUT_DIR="$5"
BANIS_REPO="/projects/weilab/liupeng/code/projects/banis"
NNUNET_REPO="/projects/weilab/liupeng/code/vendors/nnUNet"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
MODEL_FOLDER="/projects/weilab/liupeng/data/nnUNet/nnUNet_results/Dataset304_MitoEM20BANIS7TrainOnly/nnUNetTrainerBANIS7_400epochs__nnUNetBANIS7Plans__3d_fullres"
CHECKPOINT="${MODEL_FOLDER}/fold_0/checkpoint_final.pth"
INPUT_DIR="${OUTPUT_DIR}/input"
PREDICTION_DIR="${OUTPUT_DIR}/prediction"
DECODE_DIR="${OUTPUT_DIR}/decode"
EVALUATION_DIR="${OUTPUT_DIR}/evaluation"

if [[ -f "${EVALUATION_DIR}/SUCCESS" ]]; then
  echo "Existing SUCCESS: ${EVALUATION_DIR}"
  exit 0
fi
for REQUIRED in "${INPUT}" "${GROUND_TRUTH}" "${CHECKPOINT}"; do
  [[ -f "${REQUIRED}" ]] || { echo "Missing required input: ${REQUIRED}" >&2; exit 1; }
done
if [[ "${MASK}" != "-" ]]; then
  [[ -f "${MASK}" ]] || { echo "Missing mask: ${MASK}" >&2; exit 1; }
fi

mkdir -p "${INPUT_DIR}" "${PREDICTION_DIR}" "${DECODE_DIR}" "${EVALUATION_DIR}"
ln -sfn "${INPUT}" "${INPUT_DIR}/${CASE}_0000.nii.gz"

export nnUNet_raw="/projects/weilab/liupeng/data/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/projects/weilab/liupeng/data/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/projects/weilab/liupeng/data/nnUNet/nnUNet_results"
export PYTHONPATH="${NNUNET_REPO}:${BANIS_REPO}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${BANIS_REPO}"
"${PYTHON}" scripts/run_nnunet_banis7_predict.py \
  -i "${INPUT_DIR}" \
  -o "${PREDICTION_DIR}" \
  -m "${MODEL_FOLDER}" \
  -f 0 \
  -chk checkpoint_final.pth \
  -device cuda \
  --step-size 0.5 \
  --num-processes-preprocessing 1 \
  --channels 6

PREDICTION="${PREDICTION_DIR}/${CASE}.npz"
[[ -f "${PREDICTION}" ]] || { echo "Missing prediction: ${PREDICTION}" >&2; exit 1; }
DECODE_ARGS=(
  --prediction "${PREDICTION}"
  --output-dir "${DECODE_DIR}"
  --sdt-channel 0
  --seed-threshold 0.5
  --foreground-threshold 0.0
  --min-size 200
  --edt-downsample-factor 2
  --edt-parallel "${SLURM_CPUS_PER_TASK:-8}"
)
EVAL_ARGS=(
  --case "${CASE}"
  --method nnunet_banis7_sdt
  --prediction "${DECODE_DIR}/final_instance_seg.tiff"
  --ground-truth "${GROUND_TRUTH}"
  --output-dir "${EVALUATION_DIR}"
  --block-shape 32 256 256
)
if [[ "${MASK}" != "-" ]]; then
  DECODE_ARGS+=(--mask "${MASK}")
  EVAL_ARGS+=(--mask "${MASK}")
fi
"${PYTHON}" scripts/decode_banis7_sdt_prediction.py "${DECODE_ARGS[@]}"
"${PYTHON}" scripts/evaluate_instance_prediction.py "${EVAL_ARGS[@]}"
