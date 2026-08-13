#!/bin/bash
#   arch=gram_nqueens8_fp  ->  gram_nqueens10_fp
#   data_paths=[data/nqueens-8] -> [data/nqueens-10]
#
#
# steps/epoch = total_puzzles(107,480)/768 = 139.95.
#
#   RUN=nq10_fp_a3p_s0 SEED=0 EPOCHS=990 EVALINT=30 GPUS=2,3,4 PORT=29971 bash scripts/train_nqueens10_fp.sh
#   RUN=nq10_fp_a3p_s1 SEED=1 EPOCHS=990 EVALINT=30 GPUS=5,6,7 PORT=29972 bash scripts/train_nqueens10_fp.sh
#
#
# DRYRUN=1 로 조립만 확인 가능.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/venv-gram/bin/activate 2>/dev/null || true

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

LR="${LR:-1e-4}"
WD="${WD:-1.0}"
ARCH="${ARCH:-gram_nqueens10_fp}"
PELR="${PELR:-$LR}"
PEWD="${PEWD:-$WD}"

GPUS="${GPUS:-2,3,4}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29971}"
# 2026-08-03 minimal: kl_reduction=sum / y one-hot zeropad 0.573 / logvar bias 0.0 /
BETA="${BETA:-0.045}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
EPOCHS="${EPOCHS:-990}"
EVALINT="${EVALINT:-30}"
SEED="${SEED:-0}"
LRMIN="${LRMIN:-1.0}"
NODECAY_MARKED="${NODECAY_MARKED:-False}"
NODECAY_LOGVAR_BIAS="${NODECAY_LOGVAR_BIAS:-False}"
RUN="${RUN:?RUN 을 줘라}"

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "[preflight] DRYRUN=1 -- GPU 점유 검사 건너뜀 (조립만 출력)"
else
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$used" -gt 1000 ] && { echo "FATAL: GPU $g 사용 중 (${used}MiB)"; exit 1; }
  done
  echo "[preflight] arch=${ARCH} lr=${LR} wd=${WD} GPUS=${GPUS} NPROC=${NPROC} PORT=${PORT} 전부 유휴 -- launch"
fi

LAUNCH=torchrun
[ "${DRYRUN:-0}" = "1" ] && LAUNCH=echo

CUDA_VISIBLE_DEVICES=${GPUS} ${LAUNCH} \
  --nproc-per-node ${NPROC} --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} --nnodes=1 \
  pretrain.py \
  arch=${ARCH} \
  data_paths="[data/nqueens-10]" \
  'evaluators=[{name: nqueens@NQueens}]' \
  epochs=${EPOCHS} eval_interval=${EVALINT} \
  seed=${SEED} \
  lr=${LR} puzzle_emb_lr=${PELR} weight_decay=${WD} puzzle_emb_weight_decay=${PEWD} \
  lr_min_ratio=${LRMIN} \
  nodecay_marked=${NODECAY_MARKED} \
  nodecay_logvar_bias=${NODECAY_LOGVAR_BIAS} \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
