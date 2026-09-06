#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
INPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test"
OUTPUT_DIR="${INPUT_ROOT}/summary_all_cases"

cd "${REPO}"
micromamba run -n sdt python scripts/summarize_mitoem2_graph_audit.py \
  --input-root "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

cat "${OUTPUT_DIR}/SUMMARY.md"
