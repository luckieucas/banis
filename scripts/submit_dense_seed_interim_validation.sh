#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_dense_seed_interim_validation_case.sh"
SUMMARIZER="${REPO}/scripts/summarize_mitoem2_graph_audit.py"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
TRAIN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train/checkpoints"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/dense_predictor_replication/interim_step155000_validation_subset"
LOG_ROOT="${OUTPUT_ROOT}/logs"

declare -A CHECKPOINTS=(
  [1]="${TRAIN_ROOT}/mitoem2_all_sdt_mednextB_lr3e-4_s1_b2_k3_ns250000/default/checkpoints/epoch=106-step=155000.ckpt"
  [2]="${TRAIN_ROOT}/mitoem2_all_sdt_mednextB_lr3e-4_s2_b2_k3_ns250000/default/checkpoints/epoch=106-step=155000.ckpt"
)

CASES=(
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|me2-beta_train01|16,16,16|192G"
  "me2-podo|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset005_ME2-Podo|me2-podo_train02|16,16,16|128G"
  "me2-stem|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset008_ME2-Stem|me2-stem_train02|30,8,8|96G"
)

mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"

SUMMARY_JOBS=()
for SEED in 1 2; do
  CKPT="${CHECKPOINTS[${SEED}]}"
  [[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
  CASE_JOBS=()
  for ENTRY in "${CASES[@]}"; do
    IFS='|' read -r SLUG DATASET_DIR CASE SPACING MEMORY <<<"${ENTRY}"
    JOB=$(sbatch --parsable \
      --partition long \
      --gres gpu:l40s:1 \
      --cpus-per-task 8 \
      --mem "${MEMORY}" \
      --time 1-00:00:00 \
      --job-name "s${SEED}i_${CASE}" \
      --output "${LOG_ROOT}/seed${SEED}.${CASE}.%j.out" \
      --error "${LOG_ROOT}/seed${SEED}.${CASE}.%j.err" \
      "${RUNNER}" "${SEED}" "${CKPT}" "${SLUG}" "${DATASET_DIR}" \
      "${CASE}" "${SPACING}" "${OUTPUT_ROOT}")
    JOB=${JOB%%;*}
    CASE_JOBS+=("${JOB}")
    echo "seed${SEED} ${CASE}: ${JOB}"
  done

  DEPENDENCY=$(IFS=:; echo "${CASE_JOBS[*]}")
  SUMMARY_JOB=$(sbatch --parsable \
    --partition short \
    --cpus-per-task 1 \
    --mem 8G \
    --time 01:00:00 \
    --dependency "afterok:${DEPENDENCY}" \
    --job-name "s${SEED}i_summary" \
    --output "${LOG_ROOT}/seed${SEED}.summary.%j.out" \
    --error "${LOG_ROOT}/seed${SEED}.summary.%j.err" \
    --wrap "'${PYTHON}' '${SUMMARIZER}' --input-root '${OUTPUT_ROOT}/seed${SEED}' --output-dir '${OUTPUT_ROOT}/seed${SEED}/summary'")
  SUMMARY_JOB=${SUMMARY_JOB%%;*}
  SUMMARY_JOBS+=("${SUMMARY_JOB}")
  echo "seed${SEED} summary: ${SUMMARY_JOB}"
done

echo "Submitted interim validation summaries: ${SUMMARY_JOBS[*]}"
