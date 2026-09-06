#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1"
OUTPUT_ROOT="${RUN_ROOT}/downstream_morphometry"
SCORER="${RUN_ROOT}/learned_edge_scorer_stage1/models/seed0/edge_scorer.joblib"
RGMWS_ROOT="${RUN_ROOT}/region_mutex_watershed"

submit_cohort() {
  local cohort="$1"
  local expected="$2"
  local upstream_dependency="$3"
  shift 3
  local output="${OUTPUT_ROOT}/${cohort}"
  local logs="${output}/logs"
  mkdir -p "${logs}"
  local jobs=()
  local graph case lut job dependency_args=()
  if [[ -n "${upstream_dependency}" ]]; then
    dependency_args=(--dependency "afterok:${upstream_dependency}")
  fi
  for graph in "$@"; do
    case=$(jq -r .case "${graph}/config.json")
    lut="${RGMWS_ROOT}/${cohort}/by_case/${case}/label_lut_beta_0.50.npy"
    [[ -f "${lut}" ]] || { echo "Missing locked RG-MWS LUT: ${lut}" >&2; exit 1; }
    job=$(sbatch --parsable --partition short --cpus-per-task 4 --mem 64G --time 04:00:00 \
      "${dependency_args[@]}" --job-name "morph_${case}" \
      --output "${logs}/${case}.%j.out" --error "${logs}/${case}.%j.err" \
      --wrap "cd '${REPO}' && PYTHONPATH='${REPO}' '${PYTHON}' scripts/evaluate_instance_morphometry.py --graph-audit-dir '${graph}' --region-mws-lut '${lut}' --edge-scorer '${SCORER}' --output-dir '${output}/by_case/${case}' --graph-threshold 0.40 --probability-threshold 0.70 --maximum-uncertainty 1.0 --minimum-evidence 8 --minimum-affinity-mean 0.40 --minimum-high-fraction 0.50 --block-shape 32 256 256 --iou-threshold 0.50")
    job=${job%%;*}
    jobs+=("${job}")
    echo "${cohort} ${case}: ${job}"
  done
  [[ "${#jobs[@]}" -eq "${expected}" ]] || { echo "Expected ${expected} ${cohort} cases" >&2; exit 1; }
  local dependency summary
  dependency=$(IFS=:; echo "${jobs[*]}")
  summary=$(sbatch --parsable --partition short --cpus-per-task 1 --mem 8G --time 01:00:00 \
    --dependency "afterok:${dependency}" --job-name "morph_${cohort}_summary" \
    --output "${logs}/summary.%j.out" --error "${logs}/summary.%j.err" \
    --wrap "cd '${REPO}' && '${PYTHON}' scripts/summarize_instance_morphometry.py --input-root '${output}/by_case' --output-dir '${output}/summary' --expected-cases '${expected}' --cohort '${cohort}' --bootstrap-replicates 10000 --seed 0")
  LAST_SUMMARY_JOB=${summary%%;*}
  echo "${cohort} summary: ${LAST_SUMMARY_JOB}"
}

mapfile -t VALIDATION_GRAPHS < <(find "${RUN_ROOT}/heldout_validation" -type d -name graph_audit | sort)
mapfile -t TEST_GRAPHS < <(find "${RUN_ROOT}/validation_threshold_transfer_test" -type d -name graph_eval_threshold_0.40 | sort)
mapfile -t EXTERNAL_GRAPHS < <(find "${RUN_ROOT}/external_confirmatory" -type d -name graph_audit ! -path '*/kidney/*' | sort)

submit_cohort validation 11 "" "${VALIDATION_GRAPHS[@]}"
VALIDATION_SUMMARY_JOB="${LAST_SUMMARY_JOB}"
# Test and external jobs are submitted now but cannot start until the validation
# implementation sanity check has completed successfully.
submit_cohort test 11 "${VALIDATION_SUMMARY_JOB}" "${TEST_GRAPHS[@]}"
TEST_SUMMARY_JOB="${LAST_SUMMARY_JOB}"
submit_cohort external 2 "${VALIDATION_SUMMARY_JOB}" "${EXTERNAL_GRAPHS[@]}"
EXTERNAL_SUMMARY_JOB="${LAST_SUMMARY_JOB}"

echo "validation_summary=${VALIDATION_SUMMARY_JOB}"
echo "test_summary=${TEST_SUMMARY_JOB}"
echo "external_summary=${EXTERNAL_SUMMARY_JOB}"
