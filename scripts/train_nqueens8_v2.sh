#!/bin/bash
# GRAM N-Queens 8x8 v2 런처. train_nqueens8_fix.sh 의 사본 -- 원본은 건드리지 않는다.
# 2026-08-03 minimal: kl sum / y one-hot zeropad 0.573 / logvar bias 0.0 / y add /
# 기본값은 전부 현행과 동일해서 아무것도 안 주면 train_nqueens8_fix.sh 와 같은 config 다.
#
# 기존 train_nqueens8*.sh 는 건드리지 않는다. 모든 손잡이를 환경변수로 뽑아서
# 다음 arm 을 sh 수정 없이 돌릴 수 있게 한다:
#
#   RUN=gram_nq8_fix_b               bash scripts/train_nqueens8_v2.sh   # EMA off (기본)
#   RUN=gram_nq8_fix_a EMA=True      bash scripts/train_nqueens8_v2.sh   # EMA on 대조군
#   RUN=gram_nq8_fix_c BETA=0.3      bash scripts/train_nqueens8_v2.sh
#   RUN=gram_nq8_fix_d BETA=1.0      bash scripts/train_nqueens8_v2.sh
#
# DRYRUN=1 을 주면 torchrun 을 echo 로 바꿔 조립 결과만 출력하고 실행하지 않는다.
set -euo pipefail

# 원본 스크립트들은 /workspace/TinyRecursiveModels 를 하드코딩했으나 이 박스의
# repo 는 다른 경로다. 스크립트 자신의 위치를 repo 루트로 쓴다.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/ -> repo 루트

source ~/venv-gram/bin/activate 2>/dev/null || true   # bengio venv, 없으면 무시

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

GPUS="${GPUS:-3,4,5,6}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29621}"
BETA="${BETA:-0.07}"      # nq8 final 값 (sum 눈금)
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
# 큰 β 는 수렴이 느려 epochs 를 늘려야 정점을 감싼다. 기본값은 기존과 동일한 3000 이라
# EPOCHS 를 안 주면 이전 run 들과 bit-identical 하다.
EPOCHS="${EPOCHS:-3000}"
# seed 편차 측정용. pretrain 의 seed 필드(cfg_pretrain.yaml:37 seed: 0)를 덮어쓴다.
# 기본 0 = 기존과 동일.
SEED="${SEED:-0}"
# ↓↓↓ 아래 4줄은 유령 주석이다: 설명하는 환경변수가 코드에서 제거됐다 (2026-08-03 minimal).
#     DUALEVAL / LOGVARBIAS / YINJECT 는 더 이상 손잡이가 아니며 전부 하드코딩이다.
#     dual eval  -> pretrain.py:668-689  EMA 시 항상 수행 (RNG snapshot/restore 로 보호)
#     logvar bias-> gram.py:204-205      prior/post 둘 다 0.0 고정
#     y_inject   -> gram.py:324,344      u_y = z_H + y_emb (add 고정, concat 분기 삭제)
# EMA run 에서 raw(비EMA) 가중치도 매 eval 마다 같이 평가해 `_raw` 접미사로 로깅.
# eval 비용만 2배가 되고 학습 경로는 그대로다. 기본 False = 기존 동작 불변.
# (EMA=False 면 애초에 raw 를 평가하므로 이 값은 무시된다.)
# logvar bias 초기값. -4.0 = 기존 하드코딩값(초기 sigma 0.135). 0.0 이면 초기 sigma 1.0.
# posterior 의 y 주입 방식. add = z_H + y_emb (기존). concat = [z_H, y_emb] 사영.
# cosine decay 하한 비율. 1.0 = decay 없음(기존). 0.1 이면 base_lr 의 10%까지 감쇠.
LRMIN="${LRMIN:-1.0}"
# 유령 주석: NODECAY_Y_PROJ / NODECAY_LOGVAR_BIAS 는 코드에서 제거됨 -- 하드코딩.
#   현행 손잡이는 nodecay_marked 하나뿐이고, 이 런처는 CLI 에 안 넘기므로 false 고정이다.
#   (nodecay_marked 를 쓰려면 train_nqueens8_fp.sh 나 train_nqueens8_stageA.sh 를 쓸 것)
RUN="${RUN:-gram_nq8_v2_a}"

# launch 직전 점유 확인: 다른 작업이 쓰는 GPU면 즉시 중단.
# DRYRUN 은 GPU 를 안 쓰므로 이 검사를 건너뛴다(조립 확인만 하려는데 남의 작업 때문에
# 막히면 안 된다). 실제 launch 경로의 검사는 그대로다.
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
  arch=gram_nqueens8 \
  data_paths="[data/nqueens-8]" \
  'evaluators=[{name: nqueens@NQueens}]' \
  epochs=${EPOCHS} eval_interval=200 \
  seed=${SEED} \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  lr_min_ratio=${LRMIN} \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
