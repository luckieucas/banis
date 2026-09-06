#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_nnunet_banis7_sdt_case.sh"
SUMMARIZER="${REPO}/scripts/summarize_instance_predictions.py"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
RAW_ROOT="/projects/weilab/liupeng/data/raw/mito/MitoEM2.0"
GRAPH_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/validation_threshold_transfer_test"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/nnunet_banis7_matched/test_sdt"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"
chmod +x "${RUNNER}"

mapfile -t GRAPHS < <(find "${GRAPH_ROOT}" -type d -name graph_eval_threshold_0.40 | sort)
[[ "${#GRAPHS[@]}" -eq 11 ]] || { echo "Expected 11 test graphs" >&2; exit 1; }

JOBS=()
for GRAPH in "${GRAPHS[@]}"; do
  CASE=$(jq -r .case "${GRAPH}/config.json")
  GROUND_TRUTH=$(jq -r .ground_truth "${GRAPH}/config.json")
  MASK=$(jq -r .mask "${GRAPH}/config.json")
  [[ -n "${MASK}" ]] || MASK="-"
  mapfile -t INPUTS < <(find "${RAW_ROOT}" -path "*/imagesTs/${CASE}_0000.nii.gz" -type f)
  [[ "${#INPUTS[@]}" -eq 1 ]] || { echo "Expected one input for ${CASE}, found ${#INPUTS[@]}" >&2; exit 1; }

  MEMORY=96G
  TIME_LIMIT=12:00:00
  if [[ "${CASE}" == "me2-pyra_test01" ]]; then
    MEMORY=180G
    TIME_LIMIT=1-00:00:00
  fi
  JOB=$(sbatch --parsable --partition long --nodelist g013 \
    --gres gpu:l40s:1 --cpus-per-task 8 --mem "${MEMORY}" --time "${TIME_LIMIT}" \
    --job-name "nnub7_${CASE}" \
    --output "${LOG_ROOT}/${CASE}.%j.out" --error "${LOG_ROOT}/${CASE}.%j.err" \
    "${RUNNER}" "${CASE}" "${INPUTS[0]}" "${GROUND_TRUTH}" "${MASK}" \
    "${OUTPUT_ROOT}/by_case/${CASE}")
  JOB=${JOB%%;*}
  JOBS+=("${JOB}")
  echo "${CASE}: ${JOB}"
done

DEPENDENCY=$(IFS=:; echo "${JOBS[*]}")
SUMMARY_JOB=$(sbatch --parsable --partition short --cpus-per-task 1 --mem 8G --time 01:00:00 \
  --dependency "afterok:${DEPENDENCY}" --job-name nnub7_test_summary \
  --output "${LOG_ROOT}/summary.%j.out" --error "${LOG_ROOT}/summary.%j.err" \
  --wrap "'${PYTHON}' '${SUMMARIZER}' --input-root '${OUTPUT_ROOT}/by_case' --output-dir '${OUTPUT_ROOT}/summary' --expected-cases 11")
echo "summary: ${SUMMARY_JOB%%;*}"
