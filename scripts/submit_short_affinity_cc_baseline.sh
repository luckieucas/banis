#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_short_affinity_cc_case.sh"
SUMMARIZER="${REPO}/scripts/summarize_short_affinity_cc.py"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/short_affinity_cc"
LOG_ROOT="${ROOT}/logs"
SELECTION="${ROOT}/validation/summary/selection.json"
mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"

submit_case() {
  local phase="$1" graph_dir="$2" output_dir="$3" selection="$4" dependency="$5"
  local case memory=64G
  case=$(jq -r .case "${graph_dir}/config.json")
  if [[ "${case}" == "me2-pyra_test01" ]]; then memory=192G
  elif [[ "${case}" == me2-beta_* || "${case}" == me2-mossy_* || "${case}" == "me2-sperm_test01" ]]; then memory=96G
  fi
  local dependency_args=()
  [[ -n "${dependency}" ]] && dependency_args=(--dependency "afterok:${dependency}")
  sbatch --parsable --partition short --cpus-per-task 4 --mem "${memory}" --time 12:00:00 \
    "${dependency_args[@]}" --job-name "cc_${phase}_${case}" \
    --output "${LOG_ROOT}/${phase}.${case}.%j.out" --error "${LOG_ROOT}/${phase}.${case}.%j.err" \
    "${RUNNER}" "${graph_dir}" "${output_dir}" "${selection}"
}

mapfile -t VAL_GRAPHS < <(find /projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation -type d -name graph_audit | sort)
[[ "${#VAL_GRAPHS[@]}" -eq 11 ]] || { echo "Expected 11 validation graphs" >&2; exit 1; }
VAL_JOBS=()
for GRAPH in "${VAL_GRAPHS[@]}"; do
  CASE=$(jq -r .case "${GRAPH}/config.json")
  JOB=$(submit_case val "${GRAPH}" "${ROOT}/validation/by_case/${CASE}" - "")
  VAL_JOBS+=("${JOB}")
  echo "validation ${CASE}: ${JOB}"
done
VAL_DEP=$(IFS=:; echo "${VAL_JOBS[*]}")
VAL_SUMMARY=$(sbatch --parsable --partition short --cpus-per-task 1 --mem 8G --time 01:00:00 \
  --dependency "afterok:${VAL_DEP}" --job-name cc_val_summary \
  --output "${LOG_ROOT}/validation_summary.%j.out" --error "${LOG_ROOT}/validation_summary.%j.err" \
  --wrap "'${PYTHON}' '${SUMMARIZER}' --input-root '${ROOT}/validation/by_case' --output-dir '${ROOT}/validation/summary' --expected-cases 11")
echo "validation summary: ${VAL_SUMMARY}"

mapfile -t TEST_GRAPHS < <(find /projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test -type d -name graph_eval_threshold_0.40 | sort)
[[ "${#TEST_GRAPHS[@]}" -eq 11 ]] || { echo "Expected 11 test graphs" >&2; exit 1; }
TEST_JOBS=()
for GRAPH in "${TEST_GRAPHS[@]}"; do
  CASE=$(jq -r .case "${GRAPH}/config.json")
  JOB=$(submit_case test "${GRAPH}" "${ROOT}/test/by_case/${CASE}" "${SELECTION}" "${VAL_SUMMARY}")
  TEST_JOBS+=("${JOB}")
  echo "test ${CASE}: ${JOB}"
done
TEST_DEP=$(IFS=:; echo "${TEST_JOBS[*]}")
TEST_SUMMARY=$(sbatch --parsable --partition short --cpus-per-task 1 --mem 8G --time 01:00:00 \
  --dependency "afterok:${TEST_DEP}" --job-name cc_test_summary \
  --output "${LOG_ROOT}/test_summary.%j.out" --error "${LOG_ROOT}/test_summary.%j.err" \
  --wrap "'${PYTHON}' '${SUMMARIZER}' --input-root '${ROOT}/test/by_case' --output-dir '${ROOT}/test/summary' --expected-cases 11 --locked-selection '${SELECTION}'")
echo "test summary: ${TEST_SUMMARY}"

mapfile -t EXT_GRAPHS < <(find /projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/external_confirmatory -type d -name graph_audit | sort)
[[ "${#EXT_GRAPHS[@]}" -eq 3 ]] || { echo "Expected 3 external graphs" >&2; exit 1; }
EXT_JOBS=()
for GRAPH in "${EXT_GRAPHS[@]}"; do
  CASE=$(jq -r .case "${GRAPH}/config.json")
  JOB=$(submit_case ext "${GRAPH}" "${ROOT}/external/by_case/${CASE}" "${SELECTION}" "${VAL_SUMMARY}")
  EXT_JOBS+=("${JOB}")
  echo "external ${CASE}: ${JOB}"
done
EXT_DEP=$(IFS=:; echo "${EXT_JOBS[*]}")
EXT_SUMMARY=$(sbatch --parsable --partition short --cpus-per-task 1 --mem 8G --time 01:00:00 \
  --dependency "afterok:${EXT_DEP}" --job-name cc_ext_summary \
  --output "${LOG_ROOT}/external_summary.%j.out" --error "${LOG_ROOT}/external_summary.%j.err" \
  --wrap "'${PYTHON}' '${SUMMARIZER}' --input-root '${ROOT}/external/by_case' --output-dir '${ROOT}/external/summary' --expected-cases 3 --locked-selection '${SELECTION}'")
echo "external summary: ${EXT_SUMMARY}"
