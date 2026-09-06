#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_sdt_affinity_offset_ablation_case.sh"
SUMMARIZER="${REPO}/scripts/run_summarize_sdt_affinity_offset_ablation_validation.sh"
GRAPH_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/offset_ablation/validation"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${OUTPUT_ROOT}/by_case" "${LOG_ROOT}"
chmod +x "${RUNNER}" "${SUMMARIZER}"
mapfile -t GRAPH_DIRS < <(find "${GRAPH_ROOT}" -type d -name graph_audit | sort)
if [[ "${#GRAPH_DIRS[@]}" -ne 11 ]]; then
  echo "Expected 11 validation graph directories, found ${#GRAPH_DIRS[@]}" >&2
  exit 1
fi

JOBS=()
for GRAPH_DIR in "${GRAPH_DIRS[@]}"; do
  CASE=$(basename "$(dirname "${GRAPH_DIR}")")
  JOB_ID=$(sbatch \
    --parsable \
    --partition short \
    --cpus-per-task 4 \
    --mem 32G \
    --time 02:00:00 \
    --job-name "affabl_${CASE}" \
    --output "${LOG_ROOT}/${CASE}.%j.out" \
    --error "${LOG_ROOT}/${CASE}.%j.err" \
    "${RUNNER}" "${GRAPH_DIR}" "${OUTPUT_ROOT}/by_case/${CASE}")
  JOBS+=("${JOB_ID}")
  echo "Submitted ${CASE}: ${JOB_ID}"
done

DEPENDENCY=$(IFS=:; echo "${JOBS[*]}")
SUMMARY_JOB=$(sbatch \
  --parsable \
  --partition short \
  --cpus-per-task 1 \
  --mem 4G \
  --time 00:15:00 \
  --dependency "afterok:${DEPENDENCY}" \
  --job-name affabl_val_summary \
  --output "${LOG_ROOT}/summary.%j.out" \
  --error "${LOG_ROOT}/summary.%j.err" \
  "${SUMMARIZER}")

echo "Submitted dependent validation summary: ${SUMMARY_JOB}"
