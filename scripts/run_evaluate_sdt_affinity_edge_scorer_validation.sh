#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
MODEL="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1/models/seed0/edge_scorer.joblib"
GRAPH_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation"
OUTPUT_DIR="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1/validation/seed0"

mapfile -t GRAPH_DIRS < <(find "${GRAPH_ROOT}" -type d -name graph_audit | sort)
if [[ "${#GRAPH_DIRS[@]}" -ne 11 ]]; then
  echo "Expected 11 validation graph directories, found ${#GRAPH_DIRS[@]}" >&2
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
  --probability-thresholds 0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 \
  --maximum-uncertainties 0.02 0.05 0.10 0.20 1.0 \
  --minimum-evidence 8 \
  --minimum-affinity-mean 0.40 \
  --minimum-high-fraction 0.50 \
  --block-shape 32 256 256 \
  --selection-objective pooled_f1
