#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
GRAPH_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test"
OUTPUT_DIR="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/constraint_ablation/test_no_high_fraction"
mapfile -t GRAPH_DIRS < <(find "${GRAPH_ROOT}" -type d -name graph_eval_threshold_0.40 | sort)
[[ "${#GRAPH_DIRS[@]}" -eq 11 ]] || { echo "Expected 11 graph directories" >&2; exit 1; }
ARGS=()
for GRAPH_DIR in "${GRAPH_DIRS[@]}"; do ARGS+=(--graph-dir "${GRAPH_DIR}"); done

cd "${REPO}"
micromamba run -n sdt python scripts/evaluate_sdt_affinity_offset_ablation.py \
  "${ARGS[@]}" --output-dir "${OUTPUT_DIR}" \
  --fixed-setting 0.40 8 0.0 --block-shape 32 256 256
