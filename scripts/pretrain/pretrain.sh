#!/usr/bin/env bash
# Pretrain entry script (legacy: v24_exp100.sh)
set -euo pipefail

PRETRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../Pretrain" && pwd)"
port=$((1024 + RANDOM % 64512))

export OMP_NUM_THREADS=48
export CUDA_VISIBLE_DEVICES=0,1
NUM_TRAINERS=2

output_dir="/home/jiaxin/ckpts/project_01/pretrain"
data_dir="/data/jiaxin/data"

mkdir -p "${output_dir}"
cd "${PRETRAIN_DIR}"

torchrun \
    --nproc-per-node="${NUM_TRAINERS}" \
    --rdzv-endpoint="localhost:${port}" \
    main_pretrain.py \
    --data_dir "${data_dir}" \
    --batch_size 1 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.6 \
    --epochs 800 \
    --warmup_epochs 10 \
    --blr 1e-2 \
    --weight_decay 0.05 \
    --warmup_teacher_temp 0.02 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 50 \
    --freeze_last_layer 3 \
    --dist_url "tcp://localhost:${port}" \
    --output_dir "${output_dir}" \
    --log_dir "${output_dir}" \
    --accum_iter 64 \
    --mae_weight 1.0 \
    --dino_weight 1.0 \
    --out_dim 512 \
    --global_crops_scale 0.5 1 \
    --local_crops_scale 0.25 0.5 \
    --local_crops_number 8 \
    2>&1 | tee "${output_dir}/train.log"
