#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 GRAPH_DIR OUTPUT_DIR VALIDATION_SELECTION_OR_DASH" >&2
  exit 2
fi
GRAPH_DIR="$1"
OUTPUT_DIR="$2"
SELECTION="$3"
REPO="/projects/weilab/liupeng/code/projects/banis"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"

if [[ -f "${OUTPUT_DIR}/SUCCESS" ]]; then
  echo "Existing SUCCESS: ${OUTPUT_DIR}"
  exit 0
fi
ARGS=(--graph-dir "${GRAPH_DIR}" --output-dir "${OUTPUT_DIR}" --minimum-size 200 --block-shape 32 256 256)
if [[ "${SELECTION}" == "-" ]]; then
  ARGS+=(--thresholds 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90)
else
  ARGS+=(--threshold-from "${SELECTION}")
fi
cd "${REPO}"
"${PYTHON}" scripts/evaluate_short_affinity_cc.py "${ARGS[@]}"
