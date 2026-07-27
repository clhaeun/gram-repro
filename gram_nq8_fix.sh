#!/bin/bash
# Task 4: N-Queens retrain with bias-corrected EMA + live-weight saving.
# usage: gram_nq8_fix.sh <gpu> <beta> <ema:true|false> <run_name> <port>
set -euo pipefail
cd /workspace/TinyRecursiveModels

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline          # no wandb credentials on this box (sync later)
export HF_HUB_DISABLE_PROGRESS_BARS=1

gpu="$1"; beta="$2"; ema="$3"; run_name="$4"; port="$5"
mkdir -p logs

# gpu는 쉼표구분 목록 가능 ("3" 또는 "3,4,5"). nproc를 개수에서 자동 계산해
# --nproc-per-node 와 어긋나지 않게 한다 (개수 불일치로 GPU 노는 사고 방지).
NPROC="${NPROC:-$(echo $gpu | tr ',' '\n' | wc -l)}"

# launch 직전 점유 확인
for g in ${gpu//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$used" -gt 1000 ] && { echo "FATAL: GPU $g 사용 중 (${used}MiB)"; exit 1; }
done

CUDA_VISIBLE_DEVICES="${gpu}" torchrun \
  --nproc-per-node ${NPROC} --rdzv_backend=c10d --rdzv_endpoint=localhost:"${port}" --nnodes=1 \
  pretrain.py \
  arch=gram_nqueens8 \
  data_paths="[data/nqueens-8]" \
  'evaluators=[{name: nqueens@NQueens}]' \
  global_batch_size=768 \
  epochs=3000 eval_interval=200 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.kl_coef="${beta}" arch.kl_reduction=mean arch.noise_head_expansion=1.5 \
  +run_name="${run_name}" ema="${ema}" ema_rate=0.9999 \
  2>&1 | tee logs/"${run_name}".log
