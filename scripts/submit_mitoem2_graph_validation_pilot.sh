#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_mitoem2_graph_validation_case.sh"
LOG_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/heldout_validation/logs"

mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"

# One isotropic and one anisotropic held-out validation volume.  Expand only
# after these establish that thresholds transfer without looking at labelsTs.
PILOTS=(
  "me2-macro|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset003_ME2-Macro|me2-macro_train02|16,16,16|128G|1-00:00:00"
  "me2-stem|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset008_ME2-Stem|me2-stem_train02|30,8,8|96G|1-00:00:00"
)

for ENTRY in "${PILOTS[@]}"; do
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
