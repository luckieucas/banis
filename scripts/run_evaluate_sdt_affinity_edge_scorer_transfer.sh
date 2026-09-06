#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
MODEL="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1/models/seed0/edge_scorer.joblib"
VALIDATION_SUMMARY="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1/validation/seed0/evaluation_summary.json"
GRAPH_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test"
OUTPUT_DIR="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1/test_transfer/seed0"

mapfile -t GRAPH_DIRS < <(find "${GRAPH_ROOT}" -type d -name graph_eval_threshold_0.40 | sort)
if [[ "${#GRAPH_DIRS[@]}" -ne 11 ]]; then
  echo "Expected 11 test graph directories, found ${#GRAPH_DIRS[@]}" >&2
  exit 1
fi
ARGS=()
for GRAPH_DIR in "${GRAPH_DIRS[@]}"; do
  ARGS+=(--graph-dir "${GRAPH_DIR}")
done

cd "${REPO}"
micromamba run -n sdt python scripts/evaluate_sdt_affinity_edge_scorer.py \
  --model "${MODEL}" \
  "${ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --operating-point-from "${VALIDATION_SUMMARY}" \
  --minimum-evidence 8 \
  --minimum-affinity-mean 0.40 \
  --minimum-high-fraction 0.50 \
  --block-shape 32 256 256 \
  --selection-objective pooled_f1
