#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[12]$ ]]; then
  echo "Usage: $0 SEED  # SEED must be 1 or 2" >&2
  exit 2
fi

SEED="$1"
REPO="/projects/weilab/liupeng/code/projects/banis"
SAVE_ROOT="/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train/checkpoints"
EXP_NAME="mitoem2_all_sdt_mednextB_lr3e-4_s${SEED}_b2_k3_ns250000"

cd "${REPO}"
micromamba run -n sdt python BANIS.py \
  --seed "${SEED}" \
  --batch_size 2 \
  --n_steps 250000 \
  --learning_rate 0.0003 \
  --data_setting ME2_Beta ME2_Jurkat ME2_Macro ME2_Mossy ME2_Podo ME2_Pyra ME2_Sperm ME2_Stem \
  --base_data_path /projects/weilab/liupeng/data/processed/banis/mitoem2 \
  --save_path "${SAVE_ROOT}" \
  --exp_name "${EXP_NAME}" \
  --devices 1 \
  --workers 8 \
  --val_check_interval 5000 \
  --log_every_n_steps 100 \
  --model_id B \
  --wandb \
  --wandb_project banis \
  --wandb_run_name "${EXP_NAME}" \
  --wandb_group mitoem2_all_sdt_mednextB \
  --wandb_tags mitoem2 universal sdt mednextB replication \
  --no-compile \
  --no-final_full_cube_inference \
  --sdt
