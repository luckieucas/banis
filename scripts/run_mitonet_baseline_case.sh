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
MITOEM2_REPO="/projects/weilab/liupeng/code/projects/mitoem2-public"
PYTHON="/projects/weilab/liupeng/conda/envs/mitoem2/bin/python"
EVAL_PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
CONFIG="${BANIS_REPO}/configs/mitonet_v1_zero_shot.yaml"
PREDICTION_DIR="${OUTPUT_DIR}/prediction"
EVALUATION_DIR="${OUTPUT_DIR}/evaluation"

if [[ -f "${EVALUATION_DIR}/SUCCESS" ]]; then
  echo "Existing SUCCESS: ${EVALUATION_DIR}"
  exit 0
fi

mkdir -p "${PREDICTION_DIR}"
cd "${MITOEM2_REPO}"
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="${MITOEM2_REPO}:${BANIS_REPO}" "${PYTHON}" \
  "${BANIS_REPO}/scripts/run_mitonet_inference.py" \
  --config "${CONFIG}" \
  --input "${INPUT}" \
  --output "${PREDICTION_DIR}/${CASE}_mitonet_prediction.nii.gz"

PREDICTION="${PREDICTION_DIR}/${CASE}_mitonet_prediction.nii.gz"
[[ -f "${PREDICTION}" ]] || { echo "Missing prediction: ${PREDICTION}" >&2; exit 1; }

EVAL_ARGS=(
  --case "${CASE}"
  --method mitonet_v1_zero_shot
  --prediction "${PREDICTION}"
  --ground-truth "${GROUND_TRUTH}"
  --output-dir "${EVALUATION_DIR}"
  --block-shape 32 256 256
)
if [[ "${MASK}" != "-" ]]; then
  EVAL_ARGS+=(--mask "${MASK}")
fi
cd "${BANIS_REPO}"
"${EVAL_PYTHON}" scripts/evaluate_instance_prediction.py "${EVAL_ARGS[@]}"
