#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_mitoem2_graph_validation_case.sh"
LOG_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation/logs"

mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"

# Macro and Stem were completed by the pilot submission.  Pyra is intentionally
# limited to one of its four validation volumes for this first expansion.
CASES=(
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|me2-beta_train01|16,16,16|192G|1-00:00:00"
  "me2-jurkat|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset002_ME2-Jurkat|me2-jurkat_train02|16,16,16|128G|1-00:00:00"
  "me2-mossy|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset004_ME2-Mossy|me2-mossy_train03|30,8,8|192G|2-00:00:00"
  "me2-podo|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset005_ME2-Podo|me2-podo_train02|16,16,16|128G|1-00:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train01|30,8,8|384G|5-00:00:00"
  "me2-sperm|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset007_ME2-Sperm|me2-sperm_train02|16,16,16|192G|2-00:00:00"
)

for ENTRY in "${CASES[@]}"; do
  IFS='|' read -r SLUG DATASET_DIR CASE SPACING MEMORY TIME_LIMIT <<<"${ENTRY}"
  sbatch \
    --partition long \
    --gres gpu:1 \
    --cpus-per-task 8 \
    --mem "${MEMORY}" \
    --time "${TIME_LIMIT}" \
    --job-name "graphval_${SLUG}" \
    --output "${LOG_ROOT}/${SLUG}.${CASE}.%j.out" \
    --error "${LOG_ROOT}/${SLUG}.${CASE}.%j.err" \
    "${RUNNER}" "${SLUG}" "${DATASET_DIR}" "${CASE}" "${SPACING}"
done
