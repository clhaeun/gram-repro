#!/bin/bash
# GRAM N-Queens 10x10 런처. train_nqueens8_fix.sh 의 사본 -- 아래 diff 외에는 동일.
# steps/epoch = total_groups(36907)/768 = 48.06 (pretrain.py:223 은 total_groups 기준).
# EPOCHS=900, EVALINT=30 -> 총 43250 step, eval 30회, eval 간격 1442 step.
#
# 기존 train_nqueens8*.sh 는 건드리지 않는다. 모든 손잡이를 환경변수로 뽑아서
# 다음 arm 을 sh 수정 없이 돌릴 수 있게 한다:
#
#   RUN=gram_nq8_fix_b               bash train_nqueens8_fix.sh   # EMA off (기본)
#   RUN=gram_nq8_fix_a EMA=True      bash train_nqueens8_fix.sh   # EMA on 대조군
#   RUN=gram_nq8_fix_c BETA=0.3      bash train_nqueens8_fix.sh
#   RUN=gram_nq8_fix_d BETA=1.0      bash train_nqueens8_fix.sh
#
# DRYRUN=1 을 주면 torchrun 을 echo 로 바꿔 조립 결과만 출력하고 실행하지 않는다.
set -euo pipefail

# 원본 스크립트들은 /workspace/TinyRecursiveModels 를 하드코딩했으나 이 박스의
# repo 는 다른 경로다. 스크립트 자신의 위치를 repo 루트로 쓴다.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ~/venv-gram/bin/activate 2>/dev/null || true   # bengio venv, 없으면 무시

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

GPUS="${GPUS:-3,4,5,6}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29621}"
BETA="${BETA:-0.045}"      # nq10 final 값 (sum 눈금)
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
# 큰 β 는 수렴이 느려 epochs 를 늘려야 정점을 감싼다. 기본값은 기존과 동일한 3000 이라
# EPOCHS 를 안 주면 이전 run 들과 bit-identical 하다.
EPOCHS="${EPOCHS:-900}"
# seed 편차 측정용. pretrain 의 seed 필드(cfg_pretrain.yaml:37 seed: 0)를 덮어쓴다.
# 기본 0 = 기존과 동일.
SEED="${SEED:-0}"
# EMA run 에서 raw(비EMA) 가중치도 매 eval 마다 같이 평가해 `_raw` 접미사로 로깅.
# eval 비용만 2배가 되고 학습 경로는 그대로다. 기본 False = 기존 동작 불변.
# (EMA=False 면 애초에 raw 를 평가하므로 이 값은 무시된다.)
EVALINT="${EVALINT:-30}"
RUN="${RUN:-gram_nq10_a}"

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
  arch=gram_nqueens10 \
  data_paths="[data/nqueens-10]" \
  'evaluators=[{name: nqueens@NQueens}]' \
  epochs=${EPOCHS} eval_interval=${EVALINT} \
  seed=${SEED} \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
