#!/bin/bash
# 2026-08-03 minimal: kl sum / y one-hot zeropad 0.573 / logvar bias 0.0 / y add /
#
#
#   RUN=gram_nq8_fix_b               bash scripts/train_nqueens10_v2.sh   # EMA off (기본)
#   RUN=gram_nq8_fix_a EMA=True      bash scripts/train_nqueens10_v2.sh   # EMA on 대조군
#   RUN=gram_nq8_fix_c BETA=0.3      bash scripts/train_nqueens10_v2.sh
#   RUN=gram_nq8_fix_d BETA=1.0      bash scripts/train_nqueens10_v2.sh
#
# DRYRUN=1 을 주면 torchrun 을 echo 로 바꿔 조립 결과만 출력하고 실행하지 않는다.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/venv-gram/bin/activate 2>/dev/null || true

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

GPUS="${GPUS:-3,4,5,6}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29621}"
BETA="${BETA:-0.045}"
NODECAY_LOGVAR_BIAS="${NODECAY_LOGVAR_BIAS:-False}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
EPOCHS="${EPOCHS:-990}"
SEED="${SEED:-0}"
EVALINT="${EVALINT:-30}"
LRMIN="${LRMIN:-1.0}"
RUN="${RUN:-gram_nq10_a}"

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "[preflight] DRYRUN=1 -- GPU 점유 검사 건너뜀 (조립만 출력)"
else
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$used" -gt 1000 ] && { echo "FATAL: GPU $g 사용 중 (${used}MiB)"; exit 1; }
  done
  echo "[preflight] GPUS=${GPUS} NPROC=${NPROC} PORT=${PORT} 전부 유휴 -- launch"
fi

LAUNCH=torchrun
[ "${DRYRUN:-0}" = "1" ] && LAUNCH=echo

CUDA_VISIBLE_DEVICES=${GPUS} ${LAUNCH} \
  --nproc-per-node ${NPROC} --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} --nnodes=1 \
  pretrain.py \
  arch=gram_nqueens10 \
  data_paths="[data/nqueens-10]" \
  'evaluators=[{name: nqueens@NQueens}]' \
  epochs=${EPOCHS} eval_interval=${EVALINT} \
  seed=${SEED} \
  lr_min_ratio=${LRMIN} \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  nodecay_logvar_bias=${NODECAY_LOGVAR_BIAS} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
