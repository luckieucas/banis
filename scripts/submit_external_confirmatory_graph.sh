#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 LEARNED_VALIDATION_JOB_ID" >&2
  exit 2
fi

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_external_confirmatory_graph_case.sh"
SUMMARIZER="${REPO}/scripts/run_external_confirmatory_summary.sh"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/external_confirmatory"
LOG_ROOT="${RUN_ROOT}/logs"
LEARNED_VALIDATION_JOB="$1"

SPECS=(
  "cardiac|jrc_zf-cardiac-1_recon-1_test1|/projects/weilab/liupeng/data/raw/mito/Cardiac/test/jrc_zf-cardiac-1_recon-1_test1/data.zarr"
  "kidney|jrc_mus-kidney_recon-1_test1|/projects/weilab/liupeng/data/raw/mito/Kidney/test/jrc_mus-kidney_recon-1_test1/data.zarr"
  "liver|jrc_mus-liver_recon-1_test1|/projects/weilab/liupeng/data/raw/mito/Liver/test/jrc_mus-liver_recon-1_test1/data.zarr"
)

mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}" "${SUMMARIZER}"
JOBS=()
for SPEC in "${SPECS[@]}"; do
  IFS='|' read -r DOMAIN CASE DATA_ZARR <<<"${SPEC}"
  JOB_ID=$(sbatch \
    --parsable \
    --partition short \
    --gres gpu:1 \
    --cpus-per-task 8 \
    --mem 128G \
    --time 12:00:00 \
    --job-name "ext_${DOMAIN}" \
    --output "${LOG_ROOT}/${DOMAIN}.%j.out" \
    --error "${LOG_ROOT}/${DOMAIN}.%j.err" \
    "${RUNNER}" "${DOMAIN}" "${CASE}" "${DATA_ZARR}")
  JOBS+=("${JOB_ID}")
  echo "Submitted ${DOMAIN}: ${JOB_ID}"
done

DEPENDENCY=$(IFS=:; echo "${JOBS[*]}")
SUMMARY_JOB=$(sbatch \
  --parsable \
  --partition short \
  --cpus-per-task 4 \
  --mem 64G \
  --time 04:00:00 \
  --dependency "afterok:${DEPENDENCY}:${LEARNED_VALIDATION_JOB}" \
  --job-name external_confirm_summary \
  --output "${LOG_ROOT}/summary.%j.out" \
  --error "${LOG_ROOT}/summary.%j.err" \
  "${SUMMARIZER}")

echo "Submitted dependent external confirmatory summary: ${SUMMARY_JOB}"
