#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 GRAPH_DIR OUTPUT_DIR VALIDATION_SELECTION" >&2
  exit 2
fi

REPO="/projects/weilab/liupeng/code/projects/banis"
GRAPH_DIR="$1"
OUTPUT_DIR="$2"
VALIDATION_SELECTION="$3"

cd "${REPO}"
micromamba run -n sdt python scripts/evaluate_sdt_affinity_offset_ablation.py \
  --graph-dir "${GRAPH_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --settings-from "${VALIDATION_SELECTION}" \
  --block-shape 32 256 256
