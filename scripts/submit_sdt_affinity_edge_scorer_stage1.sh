#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
EDGE_RUNNER="${REPO}/scripts/run_mitoem2_graph_training_edge_case.sh"
TRAIN_RUNNER="${REPO}/scripts/run_train_sdt_affinity_edge_scorer_stage1.sh"
VALIDATION_RUNNER="${REPO}/scripts/run_evaluate_sdt_affinity_edge_scorer_validation.sh"
TEST_RUNNER="${REPO}/scripts/run_evaluate_sdt_affinity_edge_scorer_transfer.sh"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/learned_edge_scorer_stage1"
LOG_ROOT="${RUN_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
chmod +x "${EDGE_RUNNER}" "${TRAIN_RUNNER}" "${VALIDATION_RUNNER}" "${TEST_RUNNER}"

# Exact complement of the frozen 11-volume validation set within labelsTr.
SPECS=(
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|me2-beta_train02|16,16,16|192G|1-00:00:00"
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|me2-beta_train03|16,16,16|192G|1-00:00:00"
  "me2-beta|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta|me2-beta_train04|16,16,16|192G|1-00:00:00"
  "me2-jurkat|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset002_ME2-Jurkat|me2-jurkat_train01|16,16,16|128G|1-00:00:00"
  "me2-macro|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset003_ME2-Macro|me2-macro_train01|16,16,16|128G|1-00:00:00"
  "me2-mossy|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset004_ME2-Mossy|me2-mossy_train01|30,8,8|192G|2-00:00:00"
  "me2-mossy|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset004_ME2-Mossy|me2-mossy_train02|30,8,8|256G|2-00:00:00"
  "me2-podo|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset005_ME2-Podo|me2-podo_train01|16,16,16|128G|1-00:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train02|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train03|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train04|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train05|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train06|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train07|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train08|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train09|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train10|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train14|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train15|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train16|30,8,8|64G|04:00:00"
  "me2-pyra|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset006_ME2-Pyra|me2-pyra_train17|30,8,8|64G|04:00:00"
  "me2-sperm|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset007_ME2-Sperm|me2-sperm_train01|16,16,16|192G|2-00:00:00"
  "me2-stem|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset008_ME2-Stem|me2-stem_train01|30,8,8|96G|1-00:00:00"
)

EDGE_JOBS=()
for SPEC in "${SPECS[@]}"; do
  IFS='|' read -r DATASET_SLUG DATASET_DIR CASE SPACING MEMORY TIME_LIMIT <<<"${SPEC}"
  JOB_ID=$(sbatch \
    --parsable \
    --partition medium \
    --gres gpu:1 \
    --cpus-per-task 8 \
    --mem "${MEMORY}" \
    --time "${TIME_LIMIT}" \
    --job-name "edge_${CASE}" \
    --output "${LOG_ROOT}/${CASE}.%j.out" \
    --error "${LOG_ROOT}/${CASE}.%j.err" \
    "${EDGE_RUNNER}" "${DATASET_SLUG}" "${DATASET_DIR}" "${CASE}" "${SPACING}")
  EDGE_JOBS+=("${JOB_ID}")
  echo "Submitted ${CASE}: ${JOB_ID}"
done

EDGE_DEPENDENCY=$(IFS=:; echo "${EDGE_JOBS[*]}")
TRAIN_JOB=$(sbatch \
  --parsable \
  --partition short \
  --cpus-per-task 4 \
  --mem 32G \
  --time 02:00:00 \
  --dependency "afterok:${EDGE_DEPENDENCY}" \
  --job-name edge_scorer_train \
  --output "${LOG_ROOT}/train.%j.out" \
  --error "${LOG_ROOT}/train.%j.err" \
  "${TRAIN_RUNNER}")

VALIDATION_JOB=$(sbatch \
  --parsable \
  --partition short \
  --cpus-per-task 4 \
  --mem 64G \
  --time 04:00:00 \
  --dependency "afterok:${TRAIN_JOB}" \
  --job-name edge_scorer_val \
  --output "${LOG_ROOT}/validation.%j.out" \
  --error "${LOG_ROOT}/validation.%j.err" \
  "${VALIDATION_RUNNER}")

TEST_JOB=$(sbatch \
  --parsable \
  --partition short \
  --cpus-per-task 4 \
  --mem 64G \
  --time 04:00:00 \
  --dependency "afterok:${VALIDATION_JOB}" \
  --job-name edge_scorer_test \
  --output "${LOG_ROOT}/test.%j.out" \
  --error "${LOG_ROOT}/test.%j.err" \
  "${TEST_RUNNER}")

printf 'Training job: %s\nValidation job: %s\nLocked test job: %s\n' "${TRAIN_JOB}" "${VALIDATION_JOB}" "${TEST_JOB}"
