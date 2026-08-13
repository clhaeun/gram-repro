#!/bin/bash
#
#
#
#   RUN=nq8_fp_A2_s0  ARM=A2  GPUS=4 SEED=0 bash scripts/train_nqueens8_fp.sh
#   RUN=nq8_fp_A3p_s0 ARM=A3p GPUS=5 SEED=0 bash scripts/train_nqueens8_fp.sh
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

ARM="${ARM:?ARM=A1|A2|A3p 를 줘라 (A1: fp 없이 lr/wd 만, A2: lr1e-3/wd1e-3 = FPRM Table4, A3p: lr1e-4/wd1.0 = 기존 정준)}"
#   A0                          --fp----> A3p(fp arch, lr1e-4/wd1.0)
#   A1                          --fp----> A2(fp arch, lr1e-3/wd1e-3)
case "$ARM" in
  A1)  LR=1e-3; WD=1e-3; ARCH=gram_nqueens8 ;;
  A2)  LR=1e-3; WD=1e-3; ARCH=gram_nqueens8_fp ;;
  A3p) LR=1e-4; WD=1.0;  ARCH=gram_nqueens8_fp ;;
  *) echo "FATAL: 알 수 없는 ARM=$ARM"; exit 1 ;;
esac
PELR="${PELR:-$LR}"
PEWD="${PEWD:-$WD}"

GPUS="${GPUS:-4}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29671}"
# 2026-08-03 minimal: kl_reduction=sum / y one-hot zeropad 0.573 / logvar bias 0.0 /
#     y one-hot    -> gram.py:235-246    embed_scale * 0.573 * one_hot zeropad
BETA="${BETA:-0.07}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
EPOCHS="${EPOCHS:-7200}"
SEED="${SEED:-0}"
LRMIN="${LRMIN:-1.0}"
NODECAY_MARKED="${NODECAY_MARKED:-False}"
RUN="${RUN:?RUN 을 줘라}"

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "[preflight] DRYRUN=1 -- GPU 점유 검사 건너뜀 (조립만 출력)"
else
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$used" -gt 1000 ] && { echo "FATAL: GPU $g 사용 중 (${used}MiB)"; exit 1; }
  done
  echo "[preflight] ARM=${ARM} arch=${ARCH} lr=${LR} wd=${WD} GPUS=${GPUS} 전부 유휴 -- launch"
fi

LAUNCH=torchrun
[ "${DRYRUN:-0}" = "1" ] && LAUNCH=echo

CUDA_VISIBLE_DEVICES=${GPUS} ${LAUNCH} \
  --nproc-per-node ${NPROC} --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} --nnodes=1 \
  pretrain.py \
  arch=${ARCH} \
  data_paths="[data/nqueens-8]" \
  'evaluators=[{name: nqueens@NQueens}]' \
  epochs=${EPOCHS} eval_interval=200 \
  seed=${SEED} \
  lr=${LR} puzzle_emb_lr=${PELR} weight_decay=${WD} puzzle_emb_weight_decay=${PEWD} \
  lr_min_ratio=${LRMIN} \
  nodecay_marked=${NODECAY_MARKED} \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
