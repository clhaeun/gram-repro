#!/bin/bash
# GRAM N-Queens 10x10 FPRM 기여1 arm 런처.
# train_nqueens8_fp.sh 의 사본 -- 아래 diff 외에는 동일하다:
#   arch=gram_nqueens8_fp  ->  gram_nqueens10_fp
#   data_paths=[data/nqueens-8] -> [data/nqueens-10]
#   ARM 분기 삭제 (10x10 은 A3' 하나만 돈다: lr 1e-4 / wd 1.0)
#   eval_interval 하드코딩 200 -> EVALINT 환경변수 (nq10 은 30)
#   BETA 기본 0.07 -> 0.045 (nq10 final 값. yaml kl_coef 와 동일해 override 는 no-op)
#   EMA 기본 True + EPOCHS 기본 990 / LRMIN 1.0 / WARMUP 500 = 50번 baseline 과 동일
#
# 기존 train_nqueens{8,10}*.sh 는 건드리지 않는다.
#
# steps/epoch = total_puzzles(107,480)/768 = 139.95.
# EPOCHS=990, EVALINT=30 -> eval 33회, eval 간격 floor(30*139.95)=4198 step.
# 첫 eval 은 step 4198, n_eval=19220 (신 데이터 test 축).
# ★ pretrain.py 의 assert epochs % eval_interval == 0 -- 990 % 30 = 0 OK.
#
#   RUN=nq10_fp_a3p_s0 SEED=0 EPOCHS=990 EVALINT=30 GPUS=2,3,4 PORT=29971 bash train_nqueens10_fp.sh
#   RUN=nq10_fp_a3p_s1 SEED=1 EPOCHS=990 EVALINT=30 GPUS=5,6,7 PORT=29972 bash train_nqueens10_fp.sh
#
# ★ lr/wd 를 1e-3 으로 바꾸지 마라. 8x8 에서 A1(기존 블록+1e-3/1e-3)이 accuracy
#   0.0087 로 죽었고 A2(fp+1e-3/1e-3)는 NaN 이었다. GRAM 은 TRM/FPRM 에 없는 KL 항이
#   sum 눈금(초기 kl_loss ~17,000)이라 lr 10배면 KL gradient 가 폭주한다.
#   FPRM 도 Maze(Table 5)에서는 lr 1e-4 / wd 1.0 을 쓴다 -- wd 인하는 pre-norm 의
#   요구사항이 아니다.
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

# A3' 정준: lr 1e-4 / wd 1.0. 50번 baseline(nq10_repro_s0/s1)과 동일값이다.
LR="${LR:-1e-4}"
WD="${WD:-1.0}"
ARCH="${ARCH:-gram_nqueens10_fp}"
# puzzle_emb 쪽은 이 repo 관례대로 본체와 동일값을 따라간다.
PELR="${PELR:-$LR}"
PEWD="${PEWD:-$WD}"

GPUS="${GPUS:-2,3,4}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29971}"
# 2026-08-03 minimal: kl_reduction=sum / y one-hot zeropad 0.573 / logvar bias 0.0 /
# y_inject add / dual eval(ema 시) 는 코드에 하드코딩돼 손잡이가 사라졌다.
# 구 기본값(KL_RED=sum, YONEHOT=True zeropad 0.573, LOGVARBIAS=0.0, YINJECT=add,
# DUALEVAL=True)과 전부 동일하므로 launch 결과는 그대로다.
BETA="${BETA:-0.045}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
EPOCHS="${EPOCHS:-990}"
EVALINT="${EVALINT:-30}"
SEED="${SEED:-0}"
LRMIN="${LRMIN:-1.0}"
NODECAY_MARKED="${NODECAY_MARKED:-False}"   # residual_scale=fixed 면 마킹 파라미터가 없어 켜면 즉사(의도)
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
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
