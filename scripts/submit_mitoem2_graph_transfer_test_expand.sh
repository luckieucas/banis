#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_mitoem2_graph_transfer_test_case.sh"
LOG_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test/logs"
FIXED_THRESHOLD="0.40"

mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"

# Macro and Stem test transfers were completed by the pilot.  Pyra uses a
# separate 384G allocation because its test volume is 500 x 4096 x 4096.
CASES=(
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|me2-beta_test01|16,16,16|192G|1-00:00:00"
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|me2-beta_test02|16,16,16|192G|1-00:00:00"
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|me2-beta_test03|16,16,16|192G|1-00:00:00"
  "me2-jurkat|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset002_ME2-Jurkat|me2-jurkat_test01|16,16,16|128G|1-00:00:00"
  "me2-mossy|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset004_ME2-Mossy|me2-mossy_test01|30,8,8|192G|2-00:00:00"
  "me2-mossy|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset004_ME2-Mossy|me2-mossy_test02|30,8,8|256G|2-00:00:00"
  "me2-podo|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset005_ME2-Podo|me2-podo_test01|16,16,16|128G|1-00:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_test01|30,8,8|384G|5-00:00:00"
  "me2-sperm|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset007_ME2-Sperm|me2-sperm_test01|16,16,16|192G|2-00:00:00"
)

for ENTRY in "${CASES[@]}"; do
  IFS='|' read -r SLUG DATASET_DIR CASE SPACING MEMORY TIME_LIMIT <<<"${ENTRY}"
  sbatch \
    --partition long \
    --gres gpu:1 \
    --cpus-per-task 8 \
    --mem "${MEMORY}" \
    --time "${TIME_LIMIT}" \
    --job-name "graphx_${CASE}" \
    --output "${LOG_ROOT}/${SLUG}.${CASE}.%j.out" \
    --error "${LOG_ROOT}/${SLUG}.${CASE}.%j.err" \
    "${RUNNER}" "${SLUG}" "${DATASET_DIR}" "${CASE}" "${SPACING}" "${FIXED_THRESHOLD}"
done
