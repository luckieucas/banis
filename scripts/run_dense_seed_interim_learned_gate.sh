#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[12]$ ]]; then
  echo "Usage: $0 <seed>  # seed must be 1 or 2" >&2
  exit 2
fi

SEED="$1"
REPO="/projects/weilab/liupeng/code/projects/banis"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1"
INPUT_ROOT="${RUN_ROOT}/dense_predictor_replication/interim_step155000_validation_subset/seed${SEED}"
OUTPUT_DIR="${INPUT_ROOT}/learned_gate_seed0_frozen"
MODEL="${RUN_ROOT}/learned_edge_scorer_stage1/models/seed0/edge_scorer.joblib"
OPERATING_POINT="${RUN_ROOT}/learned_edge_scorer_stage1/validation/seed0/evaluation_summary.json"

mapfile -t GRAPH_DIRS < <(find "${INPUT_ROOT}" -type d -name graph_eval_threshold_0.40 | sort)
if [[ "${#GRAPH_DIRS[@]}" -ne 3 ]]; then
  echo "Expected 3 interim graph directories, found ${#GRAPH_DIRS[@]}" >&2
  exit 1
fi

ARGS=()
for GRAPH_DIR in "${GRAPH_DIRS[@]}"; do
  [[ -f "${GRAPH_DIR}/SUCCESS" ]] || { echo "Incomplete graph directory: ${GRAPH_DIR}" >&2; exit 1; }
  ARGS+=(--graph-dir "${GRAPH_DIR}")
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO}"
micromamba run -n sdt python scripts/evaluate_sdt_affinity_edge_scorer.py \
  --model "${MODEL}" \
  "${ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --operating-point-from "${OPERATING_POINT}" \
  --minimum-evidence 8 \
  --minimum-affinity-mean 0.40 \
  --minimum-high-fraction 0.50 \
  --block-shape 32 256 256 \
  --selection-objective pooled_f1

echo "Completed frozen seed-0 learned gate on dense seed ${SEED}"
