#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_sdt_affinity_offset_ablation_transfer_case.sh"
SUMMARIZER="${REPO}/scripts/run_summarize_sdt_affinity_offset_ablation_transfer.sh"
GRAPH_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/offset_ablation/test_transfer"
VALIDATION_SELECTION="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/offset_ablation/validation/summary/validation_selection.json"
LOG_ROOT="${OUTPUT_ROOT}/logs"

if [[ ! -f "${VALIDATION_SELECTION}" ]]; then
  echo "Missing validation selection: ${VALIDATION_SELECTION}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/by_case" "${LOG_ROOT}"
chmod +x "${RUNNER}" "${SUMMARIZER}"
mapfile -t GRAPH_DIRS < <(find "${GRAPH_ROOT}" -type d -name graph_eval_threshold_0.40 | sort)
if [[ "${#GRAPH_DIRS[@]}" -ne 11 ]]; then
  echo "Expected 11 test graph directories, found ${#GRAPH_DIRS[@]}" >&2
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
    --job-name "afftr_${CASE}" \
    --output "${LOG_ROOT}/${CASE}.%j.out" \
    --error "${LOG_ROOT}/${CASE}.%j.err" \
    "${RUNNER}" "${GRAPH_DIR}" "${OUTPUT_ROOT}/by_case/${CASE}" "${VALIDATION_SELECTION}")
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
  --job-name affabl_test_summary \
  --output "${LOG_ROOT}/summary.%j.out" \
  --error "${LOG_ROOT}/summary.%j.err" \
  "${SUMMARIZER}")

echo "Submitted dependent locked-transfer summary: ${SUMMARY_JOB}"
