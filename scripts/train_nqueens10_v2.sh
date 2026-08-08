#!/bin/bash
# GRAM N-Queens 10x10 런처 (pair 확장 데이터).
# steps/epoch = total_groups(107,480)/768 = 139.95 -- global_batch_size 기준이라 GPU 수와 무관.
# EPOCHS=990, EVALINT=30 -> iter당 floor(30x107480/768)=4,198 step x 33 iter = 최종 138,534.
# 2026-08-03 minimal: kl sum / y one-hot zeropad 0.573 / logvar bias 0.0 / y add /
# dual eval(ema 시) / nodecay name 분기는 코드 하드코딩 -- 해당 손잡이 제거됨.
#
# 기존 train_nqueens8*.sh 는 건드리지 않는다. 모든 손잡이를 환경변수로 뽑아서
# 다음 arm 을 sh 수정 없이 돌릴 수 있게 한다:
#
#   RUN=gram_nq8_fix_b               bash scripts/train_nqueens10_v2.sh   # EMA off (기본)
#   RUN=gram_nq8_fix_a EMA=True      bash scripts/train_nqueens10_v2.sh   # EMA on 대조군
#   RUN=gram_nq8_fix_c BETA=0.3      bash scripts/train_nqueens10_v2.sh
#   RUN=gram_nq8_fix_d BETA=1.0      bash scripts/train_nqueens10_v2.sh
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
BETA="${BETA:-0.045}"      # nq10 final 값 (sum 눈금) = arch yaml 과 동일
# scheduled sampling. 각 transition 에서 확률 SS_PROB 로 prior 를 쓴다.
# 기본 0.0 = 현행과 비트 단위 동일 -> 기존 명령줄 불변.
SS_PROB="${SS_PROB:-0.0}"
# 노이즈 head 의 weight_decay 예외 (nodecay_marked 와 독립). 기본 False = 비트 동일.
NODECAY_LOGVAR_BIAS="${NODECAY_LOGVAR_BIAS:-False}"
NODECAY_LOGVAR_HEAD="${NODECAY_LOGVAR_HEAD:-False}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
NHEXP="${NHEXP:-1.5}"
EPOCHS="${EPOCHS:-990}"    # 논문 1,000 은 1000%30!=0 이라 assert 에 걸림 -> 990 (30x33)
SEED="${SEED:-0}"
EVALINT="${EVALINT:-30}"
LRMIN="${LRMIN:-1.0}"
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
  lr_min_ratio=${LRMIN} \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=768 \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  arch.ss_prob=${SS_PROB} \
  nodecay_logvar_bias=${NODECAY_LOGVAR_BIAS} nodecay_logvar_head=${NODECAY_LOGVAR_HEAD} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
