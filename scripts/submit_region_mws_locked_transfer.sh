#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
PYTHON="/projects/weilab/liupeng/.codex/envs/affogato/bin/python"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1"
OUTPUT_ROOT="${RUN_ROOT}/region_mutex_watershed"
LOCKED_BETA="0.50"

submit_cohort() {
  local cohort="$1"
  local expected="$2"
  shift 2
  local output="${OUTPUT_ROOT}/${cohort}"
  local logs="${output}/logs"
  mkdir -p "${logs}"
  local jobs=()
  local graph case job
  for graph in "$@"; do
    case=$(jq -r .case "${graph}/config.json")
    job=$(sbatch --parsable --partition short --cpus-per-task 4 --mem 64G --time 04:00:00 \
      --job-name "rgmws_${case}" \
      --output "${logs}/${case}.%j.out" --error "${logs}/${case}.%j.err" \
      --wrap "cd '${REPO}' && PYTHONPATH='${REPO}' '${PYTHON}' scripts/evaluate_region_mutex_watershed.py --graph-audit-dir '${graph}' --output-dir '${output}/by_case/${case}' --betas '${LOCKED_BETA}'")
    job=${job%%;*}
    jobs+=("${job}")
    echo "${cohort} ${case}: ${job}"
  done
  [[ "${#jobs[@]}" -eq "${expected}" ]] || { echo "Expected ${expected} ${cohort} cases" >&2; exit 1; }
  local dependency summary
  dependency=$(IFS=:; echo "${jobs[*]}")
  summary=$(sbatch --parsable --partition short --cpus-per-task 1 --mem 8G --time 01:00:00 \
    --dependency "afterok:${dependency}" --job-name "rgmws_${cohort}_summary" \
    --output "${logs}/summary.%j.out" --error "${logs}/summary.%j.err" \
    --wrap "cd '${REPO}' && '${PYTHON}' scripts/summarize_region_mutex_watershed.py --input-root '${output}/by_case' --output-dir '${output}/summary' --expected-cases '${expected}' --locked-beta '${LOCKED_BETA}'")
  echo "${cohort} summary: ${summary%%;*}"
}

mapfile -t TEST_GRAPHS < <(find "${RUN_ROOT}/validation_threshold_transfer_test" -type d -name graph_eval_threshold_0.40 | sort)
mapfile -t EXTERNAL_GRAPHS < <(find "${RUN_ROOT}/external_confirmatory" -type d -name graph_audit ! -path '*/kidney/*' | sort)
submit_cohort test 11 "${TEST_GRAPHS[@]}"
submit_cohort external 2 "${EXTERNAL_GRAPHS[@]}"
