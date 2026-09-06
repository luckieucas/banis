#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
SCRIPT="${REPO}/scripts/render_graph_repair_qualitative.py"
RUN_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1"
SCORES="${RUN_ROOT}/learned_edge_scorer_stage1/test_transfer/seed0/edge_scores.csv"
OUTPUT_ROOT="${RUN_ROOT}/qualitative"
PAPER_FIGURES="/projects/weilab/liupeng/papers/banis_sdt_affinity_tmi/figures"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}" "${PAPER_FIGURES}" "${LOG_ROOT}"
chmod +x "${SCRIPT}"

SPECS=(
  "me2-beta_test02|${RUN_ROOT}/validation_threshold_transfer_test/me2-beta/me2-beta_test02/graph_eval_threshold_0.40|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset001_ME2-Beta/imagesTs/me2-beta_test02_0000.nii.gz"
  "me2-podo_test01|${RUN_ROOT}/validation_threshold_transfer_test/me2-podo/me2-podo_test01/graph_eval_threshold_0.40|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset005_ME2-Podo/imagesTs/me2-podo_test01_0000.nii.gz"
  "me2-stem_test01|${RUN_ROOT}/validation_threshold_transfer_test/me2-stem/me2-stem_test01/graph_eval_threshold_0.40|/projects/weilab/liupeng/data/raw/mito/MitoEM2.0/Dataset008_ME2-Stem/imagesTs/me2-stem_test01_0000.nii.gz"
)
for SPEC in "${SPECS[@]}"; do
  IFS='|' read -r CASE GRAPH_DIR IMAGE <<<"${SPEC}"
  JOB_ID=$(sbatch --parsable --partition short --cpus-per-task 4 --mem 32G --time 02:00:00 \
    --job-name "qual_${CASE}" --output "${LOG_ROOT}/${CASE}.%j.out" --error "${LOG_ROOT}/${CASE}.%j.err" \
    --wrap "cd '${REPO}' && /projects/weilab/liupeng/conda/envs/sdt/bin/python '${SCRIPT}' --graph-dir '${GRAPH_DIR}' --image '${IMAGE}' --edge-scores '${SCORES}' --output '${PAPER_FIGURES}/qualitative_${CASE}.png'")
  echo "Submitted ${CASE}: ${JOB_ID}"
done

EXTERNAL_CASE="jrc_mus-kidney_recon-1_test1"
EXTERNAL_GRAPH="${RUN_ROOT}/external_confirmatory/kidney/${EXTERNAL_CASE}/graph_audit"
EXTERNAL_IMAGE="/projects/weilab/liupeng/data/raw/mito/Kidney/test/${EXTERNAL_CASE}/data.zarr/img"
EXTERNAL_SCORES="${RUN_ROOT}/external_confirmatory/learned_gate_summary/edge_scores.csv"
JOB_ID=$(sbatch --parsable --partition short --cpus-per-task 4 --mem 32G --time 02:00:00 \
  --job-name "qual_external_kidney" --output "${LOG_ROOT}/external_kidney.%j.out" --error "${LOG_ROOT}/external_kidney.%j.err" \
  --wrap "cd '${REPO}' && /projects/weilab/liupeng/conda/envs/sdt/bin/python '${SCRIPT}' --graph-dir '${EXTERNAL_GRAPH}' --image '${EXTERNAL_IMAGE}' --edge-scores '${EXTERNAL_SCORES}' --output '${PAPER_FIGURES}/qualitative_external_kidney.png'")
echo "Submitted external kidney: ${JOB_ID}"
