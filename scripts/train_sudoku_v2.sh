#!/bin/bash
#
#
#
#
#
#
#   GPUS=0,1,2,3 bash scripts/train_sudoku_v2.sh
#   DRYRUN=1 bash scripts/train_sudoku_v2.sh      # 조립만 출력, GPU 안 씀
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/venv-gram/bin/activate 2>/dev/null || true

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29701}"

DATA="${DATA:-data/sudoku-extreme-1k-aug-1000-test10k}"
EPOCHS="${EPOCHS:-50000}"
EVALINT="${EVALINT:-2000}"
LR="${LR:-1e-4}"
WD="${WD:-1.0}"
WARMUP="${WARMUP:-2000}"
LRMIN="${LRMIN:-1.0}"
GBS="${GBS:-768}"
SEED="${SEED:-0}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"

BETA="${BETA:-0.1}"
KLBAL="${KLBAL:-0.8}"
HC="${HC:-3}"                  # arch.H_cycles
LC="${LC:-6}"                  # arch.L_cycles
LL="${LL:-2}"                  # arch.L_layers
HMS="${HMS:-16}"               # arch.halt_max_steps
MLPT="${MLPT:-True}"           # arch.mlp_t
POSENC="${POSENC:-none}"       # arch.pos_encodings
NOACT="${NOACT:-True}"         # arch.no_ACT_continue

RUN="${RUN:-gram_sudoku_v2_a}"

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "[preflight] DRYRUN=1 -- GPU 점유 검사 건너뜀 (조립만 출력)"
else
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$used" -gt 1000 ] && { echo "FATAL: GPU $g 사용 중 (${used}MiB)"; exit 1; }
  done
  echo "[preflight] GPUS=${GPUS} NPROC=${NPROC} PORT=${PORT} 전부 유휴 -- launch"
  echo "[preflight] data=${DATA} epochs=${EPOCHS} beta=${BETA}(sum) L_cycles=${LC} run=${RUN}"
fi

LAUNCH=torchrun
[ "${DRYRUN:-0}" = "1" ] && LAUNCH=echo

CUDA_VISIBLE_DEVICES=${GPUS} ${LAUNCH} \
  --nproc-per-node ${NPROC} --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} --nnodes=1 \
  pretrain.py \
  arch=gram \
  data_paths="[${DATA}]" \
  evaluators="[]" \
  epochs=${EPOCHS} eval_interval=${EVALINT} \
  seed=${SEED} \
  lr=${LR} puzzle_emb_lr=${LR} weight_decay=${WD} puzzle_emb_weight_decay=${WD} \
  lr_min_ratio=${LRMIN} lr_warmup_steps=${WARMUP} \
  global_batch_size=${GBS} \
  arch.mlp_t=${MLPT} arch.pos_encodings=${POSENC} \
  arch.L_layers=${LL} \
  arch.H_cycles=${HC} arch.L_cycles=${LC} arch.halt_max_steps=${HMS} \
  arch.kl_coef=${BETA} arch.kl_balance=${KLBAL} \
  arch.no_ACT_continue=${NOACT} \
  ema=${EMA} ema_rate=${EMA_RATE} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
