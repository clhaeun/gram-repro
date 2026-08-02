#!/bin/bash
# FPRM 기여1 arm 런처. train_nqueens8_v2.sh 의 사본 + arch=gram_nqueens8_fp + lr/wd 손잡이.
# 기존 train_nqueens8*.sh 는 건드리지 않는다.
#
# run config 두 벌 (ARM 필수 -- 안 주면 즉시 중단, 잘못된 arm 무음 launch 방지):
#   ARM=A2  : fp 블록 + FPRM Table 4 Sudoku 값   lr 1e-3 / wd 1e-3
#   ARM=A3p : fp 블록 + 기존 정준값               lr 1e-4 / wd 1.0   (A3')
#
# 근거 (사수 [8-2] 정정): wd 인하는 pre-norm 의 요구사항이 아니다 -- FPRM Table 5
# Maze 가 TRM 과 동일한 lr 1e-4 / wd 1.0 에서 pre-norm + residual scaling 을 쓴다.
# A2 는 "우리 task 가 Sudoku 이므로 Table 4 를 따름", A3' 는 Maze 선례 + 기존 확립
# 사실(전부 wd=1.0 전제)과의 비교 축 -- A0(기존 L_level, 동일 lr/wd)와 L_level
# 하나만 달라 pre-norm 단독 효과를 직접 잰다.
#
#   RUN=nq8_fp_A2_s0  ARM=A2  GPUS=4 SEED=0 bash train_nqueens8_fp.sh
#   RUN=nq8_fp_A3p_s0 ARM=A3p GPUS=5 SEED=0 bash train_nqueens8_fp.sh
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

ARM="${ARM:?ARM=A1|A2|A3p 를 줘라 (A1: fp 없이 lr/wd 만, A2: lr1e-3/wd1e-3 = FPRM Table4, A3p: lr1e-4/wd1.0 = 기존 정준)}"
# A1 은 fp 블록을 쓰지 않는다 -- arch 가 기존 gram_nqueens8 이다.
#   A0(기존 arch, lr1e-4/wd1.0) --lr/wd--> A1(기존 arch, lr1e-3/wd1e-3)
#   A0                          --fp----> A3p(fp arch, lr1e-4/wd1.0)
#   A1                          --fp----> A2(fp arch, lr1e-3/wd1e-3)
# A1 이 없으면 A2 의 변화가 pre-norm 때문인지 lr/wd 때문인지 가를 수 없다.
case "$ARM" in
  A1)  LR=1e-3; WD=1e-3; ARCH=gram_nqueens8 ;;
  A2)  LR=1e-3; WD=1e-3; ARCH=gram_nqueens8_fp ;;
  A3p) LR=1e-4; WD=1.0;  ARCH=gram_nqueens8_fp ;;
  *) echo "FATAL: 알 수 없는 ARM=$ARM"; exit 1 ;;
esac
# puzzle_emb 쪽은 이 repo 관례대로 본체와 동일값을 따라간다.
PELR="${PELR:-$LR}"
PEWD="${PEWD:-$WD}"

GPUS="${GPUS:-4}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29671}"
BETA="${BETA:-0.07}"
KL_RED="${KL_RED:-sum}"        # gram_nqueens8_fp.yaml 이 sum 이므로 기본도 sum
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
YONEHOT="${YONEHOT:-True}"
YOPROJ="${YOPROJ:-zeropad}"
YOSCALE="${YOSCALE:-0.573}"
EPOCHS="${EPOCHS:-7200}"
SEED="${SEED:-0}"
DUALEVAL="${DUALEVAL:-True}"
LOGVARBIAS="${LOGVARBIAS:-0.0}"
YINJECT="${YINJECT:-add}"
LRMIN="${LRMIN:-1.0}"
NODECAY_YPROJ="${NODECAY_YPROJ:-False}"
NODECAY_LOGVARBIAS="${NODECAY_LOGVARBIAS:-False}"
NODECAY_MARKED="${NODECAY_MARKED:-False}"   # residual_scale=fixed 면 마킹 파라미터가 없어 켜면 즉사(의도)
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
  nodecay_y_proj=${NODECAY_YPROJ} nodecay_logvar_bias=${NODECAY_LOGVARBIAS} \
  nodecay_marked=${NODECAY_MARKED} \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.kl_reduction=${KL_RED} arch.noise_head_expansion=${NHEXP} \
  arch.y_onehot=${YONEHOT} arch.y_onehot_proj=${YOPROJ} arch.y_onehot_scale=${YOSCALE} \
  arch.logvar_bias_init=${LOGVARBIAS} arch.y_inject=${YINJECT} \
  ema=${EMA} ema_rate=${EMA_RATE} dual_eval=${DUALEVAL} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
