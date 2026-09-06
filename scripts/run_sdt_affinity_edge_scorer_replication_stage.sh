#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 SEED {train|validation|test|external} [anchored|unanchored]" >&2
  exit 2
fi
SEED="$1"
STAGE="$2"
VARIANT="${3:-anchored}"
if [[ ! "${SEED}" =~ ^[12]$ ]]; then
  echo "Replication seed must be 1 or 2" >&2
  exit 2
fi
if [[ ! "${VARIANT}" =~ ^(anchored|unanchored)$ ]]; then
  echo "Variant must be anchored or unanchored" >&2
  exit 2
fi

REPO="/projects/weilab/liupeng/code/projects/banis"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1"
MODEL="${RUN_ROOT}/models/seed${SEED}/edge_scorer.joblib"
ANCHOR_ARGS=()
if [[ "${VARIANT}" == "anchored" ]]; then
  VALIDATION_ROOT="${RUN_ROOT}/validation/seed${SEED}"
  TEST_ROOT="${RUN_ROOT}/test_transfer/seed${SEED}"
  EXTERNAL_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/external_confirmatory/learned_gate_summary_seed${SEED}"
  ANCHOR_ARGS=(--minimum-affinity-mean 0.40 --minimum-high-fraction 0.50)
else
  VALIDATION_ROOT="${RUN_ROOT}/validation_unanchored/seed${SEED}"
  TEST_ROOT="${RUN_ROOT}/test_transfer_unanchored/seed${SEED}"
  EXTERNAL_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/external_confirmatory/learned_unanchored_summary_seed${SEED}"
fi

cd "${REPO}"
if [[ "${STAGE}" == "train" ]]; then
  mapfile -t TABLES < <(find "${RUN_ROOT}/training_graphs" -type f -name edge_table.csv | sort)
  [[ "${#TABLES[@]}" -eq 23 ]] || { echo "Expected 23 edge tables, found ${#TABLES[@]}" >&2; exit 1; }
  ARGS=()
  for TABLE in "${TABLES[@]}"; do ARGS+=(--edge-table "${TABLE}"); done
  micromamba run -n sdt python scripts/train_sdt_affinity_edge_scorer.py \
    "${ARGS[@]}" --output-dir "${RUN_ROOT}/models/seed${SEED}" \
    --n-estimators 25 --regularization-c 1.0 --seed "${SEED}"
elif [[ "${STAGE}" == "validation" ]]; then
  mapfile -t GRAPH_DIRS < <(find /projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation -type d -name graph_audit | sort)
  [[ "${#GRAPH_DIRS[@]}" -eq 11 ]] || { echo "Expected 11 validation graphs" >&2; exit 1; }
  ARGS=(); for DIR in "${GRAPH_DIRS[@]}"; do ARGS+=(--graph-dir "${DIR}"); done
  micromamba run -n sdt python scripts/evaluate_sdt_affinity_edge_scorer.py \
    --model "${MODEL}" "${ARGS[@]}" --output-dir "${VALIDATION_ROOT}" \
    --probability-thresholds 0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 \
    --maximum-uncertainties 0.02 0.05 0.10 0.20 1.0 --minimum-evidence 8 \
    "${ANCHOR_ARGS[@]}" --block-shape 32 256 256 --selection-objective pooled_f1
elif [[ "${STAGE}" == "test" ]]; then
  mapfile -t GRAPH_DIRS < <(find /projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test -type d -name graph_eval_threshold_0.40 | sort)
  [[ "${#GRAPH_DIRS[@]}" -eq 11 ]] || { echo "Expected 11 test graphs" >&2; exit 1; }
  ARGS=(); for DIR in "${GRAPH_DIRS[@]}"; do ARGS+=(--graph-dir "${DIR}"); done
  micromamba run -n sdt python scripts/evaluate_sdt_affinity_edge_scorer.py \
    --model "${MODEL}" "${ARGS[@]}" --output-dir "${TEST_ROOT}" \
    --operating-point-from "${VALIDATION_ROOT}/evaluation_summary.json" --minimum-evidence 8 \
    "${ANCHOR_ARGS[@]}" --block-shape 32 256 256 --selection-objective pooled_f1
elif [[ "${STAGE}" == "external" ]]; then
  mapfile -t GRAPH_DIRS < <(find /projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/external_confirmatory -type d -name graph_audit | sort)
  [[ "${#GRAPH_DIRS[@]}" -eq 3 ]] || { echo "Expected 3 external graphs" >&2; exit 1; }
  ARGS=(); for DIR in "${GRAPH_DIRS[@]}"; do ARGS+=(--graph-dir "${DIR}"); done
  micromamba run -n sdt python scripts/evaluate_sdt_affinity_edge_scorer.py \
    --model "${MODEL}" "${ARGS[@]}" --output-dir "${EXTERNAL_ROOT}" \
    --operating-point-from "${VALIDATION_ROOT}/evaluation_summary.json" --minimum-evidence 8 \
    "${ANCHOR_ARGS[@]}" --block-shape 32 256 256 --selection-objective pooled_f1
else
  echo "Unknown stage: ${STAGE}" >&2
  exit 2
fi
