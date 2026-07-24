#!/bin/bash
# N-Queens 8x8 GRAM: kl_coef (beta) sweep.
# usage: gram_nq8_sweep.sh <gpu> <beta> <run_name> <port>
set -euo pipefail
cd /workspace/TinyRecursiveModels

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export HF_HUB_DISABLE_PROGRESS_BARS=1

gpu="$1"; beta="$2"; run_name="$3"; port="$4"

mkdir -p logs

CUDA_VISIBLE_DEVICES="${gpu}" torchrun \
  --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:"${port}" --nnodes=1 \
  pretrain.py \
  arch=gram_nqueens8 \
  data_paths="[data/nqueens-8]" \
  evaluators="[]" \
  global_batch_size=768 \
  epochs=3000 eval_interval=200 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.kl_coef="${beta}" \
  arch.kl_reduction=sum \
  +run_name="${run_name}" ema=True ema_rate=0.9999 \
  2>&1 | tee logs/"${run_name}".log
