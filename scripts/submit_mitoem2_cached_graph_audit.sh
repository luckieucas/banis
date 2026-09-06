#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_mitoem2_cached_graph_audit_dataset.sh"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/cached_test_analysis"

mkdir -p "${RUN_ROOT}/logs"
chmod +x "${RUNNER}"

DATASETS=(
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|16,16,16|96G|1-00:00:00"
  "me2-jurkat|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset002_ME2-Jurkat|16,16,16|64G|12:00:00"
  "me2-macro|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset003_ME2-Macro|16,16,16|64G|12:00:00"
  "me2-mossy|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset004_ME2-Mossy|30,8,8|128G|1-00:00:00"
  "me2-podo|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset005_ME2-Podo|16,16,16|64G|12:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|30,8,8|384G|5-00:00:00"
  "me2-sperm|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset007_ME2-Sperm|16,16,16|96G|1-00:00:00"
  "me2-stem|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset008_ME2-Stem|30,8,8|48G|12:00:00"
)

for ENTRY in "${DATASETS[@]}"; do
  IFS='|' read -r SLUG DATASET_DIR SPACING MEMORY TIME_LIMIT <<<"${ENTRY}"
  sbatch \
    --partition long \
    --cpus-per-task 8 \
    --mem "${MEMORY}" \
    --time "${TIME_LIMIT}" \
    --job-name "graph1_${SLUG}" \
    --output "${RUN_ROOT}/logs/${SLUG}.%j.out" \
    --error "${RUN_ROOT}/logs/${SLUG}.%j.err" \
    "${RUNNER}" "${SLUG}" "${DATASET_DIR}" "${SPACING}"
done
