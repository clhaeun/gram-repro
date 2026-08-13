#!/bin/bash
# GRAM Sudoku-Extreme 런처 (현행 코드 기준 재학습용).
#
# 과거 run `gram_sudoku_halt_only` (scripts/archive/gram_sudoku_halt_only.sh) 의
# 설정을 그대로 옮기되, 아래 두 가지가 다르다:
#
#   1. 경로 하드코딩 제거. 아카이브 런처는 `cd /workspace/TinyRecursiveModels` 였다.
#      이 스크립트는 자기 위치 기준으로 repo 루트를 계산한다 (train_gc8_v2.sh 방식).
#   2. logvar bias init 이 다르다. 과거 run 은 -4.0 (σ_init 0.135) 하드코딩이었고,
#      현행 gram.py 는 nn.Parameter(torch.tensor(0.0)) 이라 σ_init = 1.0 이다.
#      그래서 이 run 은 과거 결과의 재현이 아니라 현행 코드 기준 재측정이다.
#
# kl_coef=0.1 은 sum 눈금이다. 환산하지 마라.
#   losses.py:90-95 가 `outputs["kl"].sum(-1).sum(-1).sum()` 으로 sum 고정이고,
#   같은 주석(L91-92)이 gram_sudoku_halt_only 를 sum 쪽 run(71.85%)으로 지목한다.
#   즉 과거 run 의 0.1 은 이미 현행과 같은 눈금이다. mean 쪽은 lmean_v2(1.8%).
#
# data_paths 는 test10k 를 쓴다.
#   test/dataset.json 의 total_groups=422786 이 배열 10000행과 불일치하지만 무해하다:
#   puzzle_dataset.py:170 `_iter_test` 가 len(dataset["inputs"]) 로 순회하고,
#   pretrain.py:228 의 total_steps 는 train metadata 만 읽는다.
#
# 총 step = epochs * total_groups * mean_puzzle_examples / global_batch_size
#         = 50000 * 1000 * 1.0 / 768 = 65,104   (과거 run 최종 step_65100 과 일치)
#
# 손잡이는 전부 환경변수. 아무것도 안 주면 위 기본값 그대로다.
#   GPUS=0,1,2,3 bash scripts/train_sudoku_v2.sh
#   DRYRUN=1 bash scripts/train_sudoku_v2.sh      # 조립만 출력, GPU 안 씀
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/ -> repo 루트

# python 경로가 서버마다 다르다 (34번 기준 venv-gram).
#     34번  ~/venv-gram/bin/activate
#     50번  /opt/conda            (as-chaeun 경유)
#     29번  ~/miniconda3/envs/gram
#   `source ... || true` 라 경로가 틀려도 조용히 넘어가고 system python 으로 돌다가
#   ModuleNotFoundError: torch 로 죽는다 -- 조용히 실패하니 먼저 확인할 것.
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

BETA="${BETA:-0.1}"            # arch.kl_coef -- sum 눈금. 위 주석 참조.
KLBAL="${KLBAL:-0.8}"
HC="${HC:-3}"                  # arch.H_cycles
LC="${LC:-6}"                  # arch.L_cycles
LL="${LL:-2}"                  # arch.L_layers
HMS="${HMS:-16}"               # arch.halt_max_steps
MLPT="${MLPT:-True}"           # arch.mlp_t
POSENC="${POSENC:-none}"       # arch.pos_encodings
NOACT="${NOACT:-True}"         # arch.no_ACT_continue

RUN="${RUN:-gram_sudoku_v2_a}"

# launch 직전 점유 확인: 다른 작업이 쓰는 GPU 면 즉시 중단.
# DRYRUN 은 GPU 를 안 쓰므로 이 검사를 건너뛴다.
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
