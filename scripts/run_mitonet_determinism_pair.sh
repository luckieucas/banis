#!/usr/bin/env bash
set -euo pipefail

REPO="/projects/weilab/liupeng/code/projects/banis"
RUNNER="${REPO}/scripts/run_mitonet_baseline_case.sh"
AUDITOR="${REPO}/scripts/audit_volume_overlap.py"
PYTHON="/projects/weilab/liupeng/conda/envs/sdt/bin/python"
CASE="jrc_zf-cardiac-1_recon-1_test1"
INPUT="/projects/weilab/liupeng/data/raw/mito/Cardiac/test/${CASE}/data.zarr/img"
GROUND_TRUTH="/projects/weilab/liupeng/data/raw/mito/Cardiac/test/${CASE}/data.zarr/seg"
OUTPUT_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/mitonet_v1_zero_shot/determinism_same_gpu"

"${RUNNER}" "${CASE}" "${INPUT}" "${GROUND_TRUTH}" - "${OUTPUT_ROOT}/run_a"
"${RUNNER}" "${CASE}" "${INPUT}" "${GROUND_TRUTH}" - "${OUTPUT_ROOT}/run_b"

"${PYTHON}" "${AUDITOR}" \
  --first "${OUTPUT_ROOT}/run_a/prediction/${CASE}_mitonet_prediction.nii.gz" \
  --second "${OUTPUT_ROOT}/run_b/prediction/${CASE}_mitonet_prediction.nii.gz" \
  --output "${OUTPUT_ROOT}/canonical_overlap_audit.json" \
  --block-depth 16

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for run in ("run_a", "run_b"):
    with (root / run / "evaluation" / "metrics.csv").open(newline="") as handle:
        rows.append(next(csv.DictReader(handle)))
ignored = {"elapsed_seconds"}
compared = {key: rows[0][key] for key in rows[0] if key not in ignored}
metric_fields_equal = all(rows[0][key] == rows[1][key] for key in compared)
audit = json.loads((root / "canonical_overlap_audit.json").read_text())
result = {
    "metric_fields_equal_excluding_elapsed": metric_fields_equal,
    "canonical_arrays_equal": audit["exactly_equal"],
    "compared_metrics": compared,
}
(root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not metric_fields_equal or not audit["exactly_equal"]:
    raise SystemExit("MitoNet same-GPU determinism invariant failed")
PY
