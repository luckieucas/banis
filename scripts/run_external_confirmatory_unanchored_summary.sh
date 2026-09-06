#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/external_confirmatory"
MODEL="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1/models/seed0/edge_scorer.joblib"
VALIDATION_SUMMARY="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1/validation_unanchored/seed0/evaluation_summary.json"

mapfile -t GRAPH_DIRS < <(find "${RUN_ROOT}" -type d -name graph_audit | sort)
if [[ "${#GRAPH_DIRS[@]}" -ne 3 ]]; then
  echo "Expected 3 external graph directories, found ${#GRAPH_DIRS[@]}" >&2
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
  --output-dir "${RUN_ROOT}/learned_unanchored_summary" \
  --operating-point-from "${VALIDATION_SUMMARY}" \
  --minimum-evidence 8 \
  --block-shape 32 256 256 \
  --selection-objective pooled_f1
