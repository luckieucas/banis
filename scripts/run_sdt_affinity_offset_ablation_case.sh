#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 GRAPH_DIR OUTPUT_DIR" >&2
  exit 2
fi

REPO="/projects/weilab/liupeng/code/projects/banis"
GRAPH_DIR="$1"
OUTPUT_DIR="$2"

cd "${REPO}"
micromamba run -n sdt python scripts/evaluate_sdt_affinity_offset_ablation.py \
  --graph-dir "${GRAPH_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --fixed-setting 0.30 8 0.5 \
  --fixed-setting 0.35 8 0.5 \
  --fixed-setting 0.40 8 0.5 \
  --fixed-setting 0.45 8 0.5 \
  --fixed-setting 0.50 8 0.5 \
  --fixed-setting 0.55 8 0.5 \
  --fixed-setting 0.60 8 0.5 \
  --fixed-setting 0.40 1 0.5 \
  --fixed-setting 0.40 32 0.5 \
  --fixed-setting 0.40 8 0.0 \
  --block-shape 32 256 256
