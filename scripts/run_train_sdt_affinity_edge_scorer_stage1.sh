#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1"
TABLE_ROOT="${RUN_ROOT}/training_graphs"
OUTPUT_DIR="${RUN_ROOT}/models/seed0"

mapfile -t TABLES < <(find "${TABLE_ROOT}" -type f -name edge_table.csv | sort)
if [[ "${#TABLES[@]}" -ne 23 ]]; then
  echo "Expected 23 completed training edge tables, found ${#TABLES[@]}" >&2
  exit 1
fi
ARGS=()
for TABLE in "${TABLES[@]}"; do
  ARGS+=(--edge-table "${TABLE}")
done

cd "${REPO}"
micromamba run -n sdt python scripts/train_sdt_affinity_edge_scorer.py \
  "${ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-estimators 25 \
  --regularization-c 1.0 \
  --seed 0
