#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
PYTHON="/projects/weilab/liupeng/.codex/envs/affogato/bin/python"
INPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/region_mutex_watershed/validation"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

mapfile -t GRAPHS < <(find "${INPUT_ROOT}" -type d -name graph_audit | sort)
[[ "${#GRAPHS[@]}" -eq 11 ]] || { echo "Expected 11 validation graph audits, found ${#GRAPHS[@]}" >&2; exit 1; }

JOBS=()
for GRAPH in "${GRAPHS[@]}"; do
  CASE=$(jq -r .case "${GRAPH}/config.json")
  JOB=$(sbatch --parsable --partition short --cpus-per-task 4 --mem 64G --time 04:00:00 \
    --job-name "rgmws_${CASE}" \
    --output "${LOG_ROOT}/${CASE}.%j.out" --error "${LOG_ROOT}/${CASE}.%j.err" \
    --wrap "cd '${REPO}' && PYTHONPATH='${REPO}' '${PYTHON}' scripts/evaluate_region_mutex_watershed.py --graph-audit-dir '${GRAPH}' --output-dir '${OUTPUT_ROOT}/by_case/${CASE}'")
  JOB=${JOB%%;*}
  JOBS+=("${JOB}")
  echo "${CASE}: ${JOB}"
done

DEPENDENCY=$(IFS=:; echo "${JOBS[*]}")
SUMMARY=$(sbatch --parsable --partition short --cpus-per-task 1 --mem 8G --time 01:00:00 \
  --dependency "afterok:${DEPENDENCY}" --job-name rgmws_val_summary \
  --output "${LOG_ROOT}/summary.%j.out" --error "${LOG_ROOT}/summary.%j.err" \
  --wrap "cd '${REPO}' && '${PYTHON}' scripts/summarize_region_mutex_watershed.py --input-root '${OUTPUT_ROOT}/by_case' --output-dir '${OUTPUT_ROOT}/summary' --expected-cases 11")
echo "summary: ${SUMMARY%%;*}"
