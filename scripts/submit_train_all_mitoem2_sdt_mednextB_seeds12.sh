#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_train_all_mitoem2_sdt_mednextB_seed.sh"
LOG_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train/logs"

mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"
for SEED in 1 2; do
  JOB_ID=$(sbatch \
    --parsable \
    --partition long \
    --gres gpu:1 \
    --cpus-per-task 12 \
    --mem 192G \
    --time 5-00:00:00 \
    --job-name "mitoem2_B_sdt_s${SEED}" \
    --output "${LOG_ROOT}/mitoem2_B_sdt_s${SEED}.%j.out" \
    --error "${LOG_ROOT}/mitoem2_B_sdt_s${SEED}.%j.err" \
    "${RUNNER}" "${SEED}")
  echo "Submitted seed ${SEED}: ${JOB_ID}"
done
