#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_sdt_affinity_edge_scorer_replication_stage.sh"
LOG_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1/logs"
mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"

submit_stage() {
  local name="$1" dependency="$2" seed="$3" stage="$4" variant="$5" memory="$6" time_limit="$7"
  local dependency_args=()
  [[ -n "${dependency}" ]] && dependency_args=(--dependency "afterok:${dependency}")
  sbatch --parsable --partition short --cpus-per-task 4 --mem "${memory}" --time "${time_limit}" \
    "${dependency_args[@]}" --job-name "${name}" \
    --output "${LOG_ROOT}/${name}.%j.out" --error "${LOG_ROOT}/${name}.%j.err" \
    "${RUNNER}" "${seed}" "${stage}" "${variant}"
}

for SEED in 1 2; do
  TRAIN=$(submit_stage "edge_train_s${SEED}" "" "${SEED}" train anchored 32G 02:00:00)
  VAL_A=$(submit_stage "edge_val_a_s${SEED}" "${TRAIN}" "${SEED}" validation anchored 64G 04:00:00)
  VAL_U=$(submit_stage "edge_val_u_s${SEED}" "${TRAIN}" "${SEED}" validation unanchored 64G 04:00:00)
  TEST_A=$(submit_stage "edge_test_a_s${SEED}" "${VAL_A}" "${SEED}" test anchored 64G 04:00:00)
  TEST_U=$(submit_stage "edge_test_u_s${SEED}" "${VAL_U}" "${SEED}" test unanchored 64G 04:00:00)
  EXT_A=$(submit_stage "edge_ext_a_s${SEED}" "${VAL_A}" "${SEED}" external anchored 64G 04:00:00)
  EXT_U=$(submit_stage "edge_ext_u_s${SEED}" "${VAL_U}" "${SEED}" external unanchored 64G 04:00:00)
  printf 'seed=%s train=%s val_a=%s val_u=%s test_a=%s test_u=%s ext_a=%s ext_u=%s\n' \
    "${SEED}" "${TRAIN}" "${VAL_A}" "${VAL_U}" "${TEST_A}" "${TEST_U}" "${EXT_A}" "${EXT_U}"
done
