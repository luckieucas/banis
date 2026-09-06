#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_mitonet_baseline_case.sh"
SUMMARIZER="${REPO}/scripts/summarize_instance_predictions.py"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/mitonet_v1_zero_shot/external_nonoverlap_deterministic_v100_seed0"
LOG_ROOT="${OUTPUT_ROOT}/../logs"
mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"

CASES=(
  jrc_zf-cardiac-1_recon-1_test1
  jrc_mus-liver_recon-1_test1
)
INPUTS=(
  /projects/weilab/liupeng/data/raw/mito/Cardiac/test/jrc_zf-cardiac-1_recon-1_test1/data.zarr/img
  /projects/weilab/liupeng/data/raw/mito/Liver/test/jrc_mus-liver_recon-1_test1/data.zarr/img
)
GROUND_TRUTHS=(
  /projects/weilab/liupeng/data/raw/mito/Cardiac/test/jrc_zf-cardiac-1_recon-1_test1/data.zarr/seg
  /projects/weilab/liupeng/data/raw/mito/Liver/test/jrc_mus-liver_recon-1_test1/data.zarr/seg
)

JOBS=()
for INDEX in "${!CASES[@]}"; do
  CASE="${CASES[$INDEX]}"
  INPUT="${INPUTS[$INDEX]}"
  GROUND_TRUTH="${GROUND_TRUTHS[$INDEX]}"
  [[ -d "${INPUT}" && -d "${GROUND_TRUTH}" ]] || {
    echo "Missing external input or ground truth for ${CASE}" >&2
    exit 1
  }
  JOB=$(sbatch --parsable --partition short --nodelist g002 \
    --gres gpu:v100:1 --cpus-per-task 8 \
    --mem 96G --time 12:00:00 --job-name "mitonet_${CASE}" \
    --output "${LOG_ROOT}/${CASE}.%j.out" --error "${LOG_ROOT}/${CASE}.%j.err" \
    "${RUNNER}" "${CASE}" "${INPUT}" "${GROUND_TRUTH}" - \
    "${OUTPUT_ROOT}/by_case/${CASE}")
  JOB=${JOB%%;*}
  JOBS+=("${JOB}")
  echo "${CASE}: ${JOB}"
done

DEPENDENCY=$(IFS=:; echo "${JOBS[*]}")
SUMMARY_JOB=$(sbatch --parsable --partition short --cpus-per-task 1 --mem 8G --time 01:00:00 \
  --dependency "afterok:${DEPENDENCY}" --job-name mitonet_external_summary \
  --output "${LOG_ROOT}/external_summary.%j.out" --error "${LOG_ROOT}/external_summary.%j.err" \
  --wrap "'${PYTHON}' '${SUMMARIZER}' --input-root '${OUTPUT_ROOT}/by_case' --output-dir '${OUTPUT_ROOT}/summary' --expected-cases 2")
echo "summary: ${SUMMARY_JOB%%;*}"
