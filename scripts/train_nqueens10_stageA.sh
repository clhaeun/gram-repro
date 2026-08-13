#!/bin/bash
# Stage A (input-independent α 탐색) nq10 런처.
# nq8 판(train_nqueens8_stageA.sh)과 차이는 4곳:
#   ARCH gram_nqueens10_fp / data_paths data/nqueens-10 / BETA 0.045 / EPOCHS 990 EVALINT 30
#
#   RUN=nq10_A3p_ii_a7525 RESIDUAL_SCALE=input-independent ALPHA1=0.75 ALPHA2=0.25 \
#     NODECAY_MARKED=True EPOCHS=990 EVALINT=30 SEED=0 GPUS=6,7 PORT=29933 \
#     bash scripts/train_nqueens10_stageA.sh
#
# PORT 기본값이 nq8 판과 같은 29981 이라 동시 기동 시 반드시 명시할 것.
# RESIDUAL_SCALE = {none, fixed, input-independent, input-dependent}.  그 외는 ValueError.
# NODECAY_MARKED 는 residual_scale 에 맞춰야 한다:
#   input-independent -> True   (False 면 α 가 wd 1.0 을 맞아 α1=α2=0.5 로 붕괴한다)
#   그 외             -> False  (마킹이 0개라 True 면 pretrain.py 가 ValueError 로 즉사)
# outer_only=true 면 마킹이 2개가 아니라 1개다.  기동 로그의 `[nodecay] ... N개` 로 확인.
#
# python 경로는 서버마다 다르다.  `source ... || true` 라 경로가 틀려도 조용히 넘어가고
# system python 으로 돌다가 ModuleNotFoundError: torch 로 죽으니 먼저 확인할 것.
#
# DRYRUN=1 로 조립만 확인 가능.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/ -> repo 루트

source ~/venv-gram/bin/activate 2>/dev/null || true

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

# A3' 정준 (확정 설정) -- 바꾸지 마라.
LR="${LR:-1e-4}"
WD="${WD:-1.0}"
ARCH="${ARCH:-gram_nqueens10_fp}"
PELR="${PELR:-$LR}"
PEWD="${PEWD:-$WD}"

# Stage A 손잡이
RESIDUAL_SCALE="${RESIDUAL_SCALE:?RESIDUAL_SCALE 을 줘라 (none|fixed|input-independent|input-dependent)}"
ALPHA1="${ALPHA1:?ALPHA1 을 줘라 (0,1) 개구간}"
ALPHA2="${ALPHA2:?ALPHA2 를 줘라 (0,1) 개구간}"
# ε 주입 계수. 기본 1.0 = 현행과 비트 단위 동일 -> 기존 명령줄 불변.
# ALPHA1/2 의 ${VAR:?} 필수 스타일과 다른 것은 의도적이다(회귀 명령줄 보존).
EPS_SCALE="${EPS_SCALE:-1.0}"
# inner z_L loop damped update. 기본 1.0 = 현행과 비트 단위 동일.
DAMPING_ETA="${DAMPING_ETA:-1.0}"
# scheduled sampling. 학습 중 각 transition 에서 확률 SS_PROB 로 prior 를 쓴다.
# 기본 0.0 = 현행과 비트 단위 동일 -> 기존 명령줄 불변.
SS_PROB="${SS_PROB:-0.0}"
NODECAY_MARKED="${NODECAY_MARKED:-True}"
# 노이즈 head 의 weight_decay 예외 (nodecay_marked 와 독립). 기본 False = 비트 동일.
NODECAY_LOGVAR_BIAS="${NODECAY_LOGVAR_BIAS:-False}"
NODECAY_LOGVAR_HEAD="${NODECAY_LOGVAR_HEAD:-False}"

GPUS="${GPUS:-0}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29981}"
BETA="${BETA:-0.045}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
EPOCHS="${EPOCHS:-990}"
EVALINT="${EVALINT:-30}"
SEED="${SEED:-0}"
LRMIN="${LRMIN:-1.0}"
RUN="${RUN:?RUN 을 줘라}"

# steps/epoch = total_groups(107,480)/768 = 139.9479.  EPOCHS=990, EVALINT=30 ->
# eval 33회, 간격 floor(30*139.9479)=4198 step, 총 138,534 step.
# pretrain.py 의 assert epochs % eval_interval == 0 -- 990 % 30 = 0 OK.

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "[preflight] DRYRUN=1 -- GPU 점유 검사 건너뜀 (조립만 출력)"
else
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$used" -gt 1000 ] && { echo "FATAL: GPU $g 사용 중 (${used}MiB)"; exit 1; }
  done
  echo "[preflight] arch=${ARCH} residual_scale=${RESIDUAL_SCALE} a1=${ALPHA1} a2=${ALPHA2} nodecay=${NODECAY_MARKED} lr=${LR} wd=${WD} GPUS=${GPUS} -- launch"
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
  nodecay_logvar_bias=${NODECAY_LOGVAR_BIAS} nodecay_logvar_head=${NODECAY_LOGVAR_HEAD} \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  arch.residual_scale=${RESIDUAL_SCALE} \
  arch.alpha_1_init=${ALPHA1} arch.alpha_2_init=${ALPHA2} \
  arch.eps_scale=${EPS_SCALE} arch.damping_eta=${DAMPING_ETA} \
  arch.ss_prob=${SS_PROB} arch.ss_seed=${SEED} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
