#!/bin/bash
# Stage A (input-independent α 탐색) 전용 런처.
# train_nqueens8_fp.sh 의 A3p arm 사본 + residual_scale / α init 손잡이.
#
# ★ train_nqueens8_fp.sh 는 건드리지 않는다 -- 50번이 A0 앵커와 fixed 셀을
#   그 스크립트로 돌리는 중이다.
#
# 목적: "기여 1이 GRAM 에 전이되는 조건이 있는가."  A0 튜닝은 종료됐고 A3' 설정만 탐색한다.
# nq8 은 A0 가 99.4% 로 포화, coverage 상한 1.0000(최대 해 18 < 20) -- 필터 플랫폼이다.
# 결론은 nq10 에서 낸다.
#
#   RUN=nq8_A3p_ii_a7525_s0 RESIDUAL_SCALE=input-independent ALPHA1=0.75 ALPHA2=0.25 \
#     NODECAY_MARKED=True GPUS=0 PORT=29981 bash train_nqueens8_stageA.sh
#
# ★ residual_scale 화이트리스트 = {none, fixed, input-independent, input-dependent}
#   (models/fp_transformer.py:147).  learnable 은 없다 -- 넣으면 ValueError.
#
# ★ NODECAY_MARKED=True 를 반드시 줘라.  런처 기본값은 False 이고 pretrain.py 의
#   _adamw_param_arg 가 False 면 마킹을 통째로 무시해 α 가 wd 1.0 을 정면으로 맞는다.
#   28,095 step 이면 α logit 이 6% 로 줄어 α1=α2=0.5 로 붕괴하고, β1/β2 가 α 에서
#   유도되므로 세 셀이 전부 같은 실험이 된다.
#   여기서는 기본값을 True 로 둬서 사고를 막는다 (fixed 로 돌리면 마킹 0개라 즉사 -- 의도).
#
# ★ 반대로 RESIDUAL_SCALE=fixed 로 돌릴 때는 NODECAY_MARKED=False 를 명시해야 한다.
#   이 런처의 기본값은 True 인데 fixed 는 α 가 requires_grad=False 라 _no_weight_decay
#   마킹이 0개다 -> pretrain.py:137-138 이 ValueError 로 즉사한다.
#   (none / input-dependent 도 마킹 0개라 마찬가지로 False 를 줘야 한다.
#    input-dependent 의 α 는 CastedLinear 의 weight/bias 라 애초에 마킹 대상이 아니다.)
#
# ★ python 경로가 서버마다 다르다.  이 스크립트는 `source ~/venv-gram/bin/activate`
#   (34번 기준)를 쓴다.  다른 서버에서 돌릴 때는 그 줄을 그 서버 값으로 바꿔라:
#     34번  ~/venv-gram/bin/activate
#     50번  /opt/conda            (as-chaeun 경유)
#     29번  ~/miniconda3/envs/gram
#   `source ... || true` 라 경로가 틀려도 조용히 넘어가고 system python 으로 돌다가
#   ModuleNotFoundError: torch 로 죽는다 -- 조용히 실패하니 먼저 확인할 것.
#
# ★ outer_only=true 면 alpha_1_param 이 생성되지 않아 마킹이 2개가 아니라 1개가 된다
#   (fp_transformer.py:169-177 의 `if not self.outer_only` 분기).  기동 로그의
#   `[nodecay] weight_decay=0 대상 N개` 로 확인할 것.  기본값 false 를 유지한다.
#
# DRYRUN=1 로 조립만 확인 가능.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ~/venv-gram/bin/activate 2>/dev/null || true

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

# A3' 정준 (확정 설정) -- 바꾸지 마라.
LR="${LR:-1e-4}"
WD="${WD:-1.0}"
ARCH="${ARCH:-gram_nqueens8_fp}"
PELR="${PELR:-$LR}"
PEWD="${PEWD:-$WD}"

# Stage A 손잡이
RESIDUAL_SCALE="${RESIDUAL_SCALE:?RESIDUAL_SCALE 을 줘라 (none|fixed|input-independent|input-dependent)}"
ALPHA1="${ALPHA1:?ALPHA1 을 줘라 (0,1) 개구간}"
ALPHA2="${ALPHA2:?ALPHA2 를 줘라 (0,1) 개구간}"
NODECAY_MARKED="${NODECAY_MARKED:-True}"

GPUS="${GPUS:-0}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29981}"
BETA="${BETA:-0.07}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
EPOCHS="${EPOCHS:-3000}"
EVALINT="${EVALINT:-200}"
SEED="${SEED:-0}"
LRMIN="${LRMIN:-1.0}"
RUN="${RUN:?RUN 을 줘라}"

# steps/epoch = total_groups(7,194)/768 = 9.37.  EPOCHS=3000, EVALINT=200 ->
# eval 15회, 간격 floor(200*9.367)=1873 step, 총 28,095 step.
# ★ pretrain.py 의 assert epochs % eval_interval == 0 -- 3000 % 200 = 0 OK.

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
  data_paths="[data/nqueens-8]" \
  'evaluators=[{name: nqueens@NQueens}]' \
  epochs=${EPOCHS} eval_interval=${EVALINT} \
  seed=${SEED} \
  lr=${LR} puzzle_emb_lr=${PELR} weight_decay=${WD} puzzle_emb_weight_decay=${PEWD} \
  lr_min_ratio=${LRMIN} \
  nodecay_marked=${NODECAY_MARKED} \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  arch.residual_scale=${RESIDUAL_SCALE} \
  arch.alpha_1_init=${ALPHA1} arch.alpha_2_init=${ALPHA2} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
