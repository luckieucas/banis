#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_mitoem2_graph_validation_case.sh"
SUMMARIZER="${REPO}/scripts/run_summarize_mitoem2_graph_validation_full.sh"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation"
LOG_ROOT="${RUN_ROOT}/logs"
DATASET_DIR="/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra"

mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}" "${SUMMARIZER}"

# train01 is already complete. These are the three remaining validation volumes
# in the official local split. Each is 500 x 512 x 512 and used about 6 GB RSS
# in the completed train01 pilot, so 64 GB leaves ample headroom.
JOBS=()
for CASE in me2-pyra_train11 me2-pyra_train12 me2-pyra_train13; do
  JOB_ID=$(sbatch \
    --parsable \
    --partition short \
    --gres gpu:1 \
    --cpus-per-task 8 \
    --mem 64G \
    --time 02:00:00 \
    --job-name "graphval_${CASE}" \
    --output "${LOG_ROOT}/me2-pyra.${CASE}.%j.out" \
    --error "${LOG_ROOT}/me2-pyra.${CASE}.%j.err" \
    "${RUNNER}" me2-pyra "${DATASET_DIR}" "${CASE}" 30,8,8)
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
  --job-name graphval_full_summary \
  --output "${LOG_ROOT}/full_validation_summary.%j.out" \
  --error "${LOG_ROOT}/full_validation_summary.%j.err" \
  "${SUMMARIZER}")

echo "Submitted dependent full-validation summary: ${SUMMARY_JOB}"
