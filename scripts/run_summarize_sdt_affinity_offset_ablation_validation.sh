#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
INPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/offset_ablation/validation"
OUTPUT_DIR="${INPUT_ROOT}/summary"

cd "${REPO}"
micromamba run -n sdt python scripts/summarize_sdt_affinity_offset_ablation.py \
  --input-root "${INPUT_ROOT}/by_case" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-cases 11
