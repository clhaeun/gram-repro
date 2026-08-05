#!/bin/bash
set -euo pipefail
cd /workspace/TinyRecursiveModels

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs
run_name="gram_nq8_mean1"
beta=0.07
kl_reduction=mean
run_name="gram_nq8_mean_nh512"

# GPU 선택은 환경변수로 오버라이드 (pull마다 하드코딩 충돌 방지).
#   GPUS   : 쉼표구분 GPU 목록 (기본 0,1,2,3)
#   NPROC  : GPU 개수 자동 계산 -> --nproc-per-node 와 반드시 일치
#   PORT   : rdzv 포트 (기본 29611)
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29611}"

# launch 직전 점유 확인: 다른 작업이 쓰는 GPU면 즉시 중단
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$used" -gt 1000 ] && { echo "FATAL: GPU $g 사용 중 (${used}MiB)"; exit 1; }
done

CUDA_VISIBLE_DEVICES=${GPUS} torchrun \
  --nproc-per-node ${NPROC} --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} --nnodes=1 \
  pretrain.py \
  arch=gram_nqueens8 \
  data_paths="[data/nqueens-8]" \
  evaluators="[]" \
  epochs=3000 eval_interval=200 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  arch.kl_coef=${beta} arch.kl_reduction=${kl_reduction} arch.noise_head_expansion=1.5 \
  +run_name=${run_name} ema=True ema_rate=0.9999 \
  2>&1 | tee logs/${run_name}.log