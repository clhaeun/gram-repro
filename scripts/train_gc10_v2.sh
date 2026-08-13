#!/bin/bash
# GRAM Graph-Coloring 10-vertex 런처.
# scripts/archive/train_gc8_v2.sh 의 사본 -- 원본은 건드리지 않는다.
#
# gc8 과 다른 점은 셋뿐이다:
#   1. data_paths = data/graph-coloring-10
#   2. kl_coef = 0.45 (논문 B.2 GC 10-vertex. 0.5 가 아니다 -- 0.5 는 8-vertex)
#   3. 총 step = 5000 * 13461 / 768 = 87,637
#
# arch 는 gram_gc8 을 그대로 쓴다. seq_len / vocab_size 는 yaml 에 없고
# pretrain.py:create_model() 이 데이터 metadata 에서 주입하므로 (gram_gc8.yaml:32-34)
# GC-10 의 seq_len=45 / vocab_size=6 이 자동으로 들어간다.
#
# kl_coef 는 sum 눈금이다. 환산하지 마라 -- 논문의 N-Queens 값 0.07(8x8) /
#   0.045(10x10) 이 우리 run 과 정확히 일치하므로 우리 눈금 = 논문 눈금이다.
#
#   GPUS=4,5,6,7 bash scripts/train_gc10_v2.sh
#   DRYRUN=1 bash scripts/train_gc10_v2.sh      # 조립만 출력, GPU 안 씀
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/ -> repo 루트

source ~/venv-gram/bin/activate 2>/dev/null || true

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs

GPUS="${GPUS:-4,5,6,7}"
NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"
PORT="${PORT:-29813}"

DATA="${DATA:-data/graph-coloring-10}"
BETA="${BETA:-0.45}"          # arch.kl_coef -- 논문 B.2 GC 10-vertex (sum 눈금)
EPOCHS="${EPOCHS:-5000}"
EVALINT="${EVALINT:-250}"
SEED="${SEED:-0}"
EMA="${EMA:-True}"
EMA_RATE="${EMA_RATE:-0.9999}"
WARMUP="${WARMUP:-500}"
LRMIN="${LRMIN:-1.0}"
NHEXP="${NHEXP:-1.5}"
GBS="${GBS:-768}"

RUN="${RUN:-gc10_paper_34}"

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "[preflight] DRYRUN=1 -- GPU 점유 검사 건너뜀 (조립만 출력)"
else
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$used" -gt 1000 ] && { echo "FATAL: GPU $g 사용 중 (${used}MiB)"; exit 1; }
  done
  echo "[preflight] GPUS=${GPUS} NPROC=${NPROC} PORT=${PORT} 전부 유휴 -- launch"
  echo "[preflight] data=${DATA} epochs=${EPOCHS} beta=${BETA}(sum) run=${RUN}"
fi

LAUNCH=torchrun
[ "${DRYRUN:-0}" = "1" ] && LAUNCH=echo

CUDA_VISIBLE_DEVICES=${GPUS} ${LAUNCH} \
  --nproc-per-node ${NPROC} --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} --nnodes=1 \
  pretrain.py \
  arch=gram_gc8 \
  data_paths="[${DATA}]" \
  'evaluators=[{name: graph_coloring@GraphColoring}]' \
  epochs=${EPOCHS} eval_interval=${EVALINT} \
  seed=${SEED} \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  lr_min_ratio=${LRMIN} \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 \
  global_batch_size=${GBS} \
  arch.kl_coef=${BETA} arch.noise_head_expansion=${NHEXP} \
  ema=${EMA} ema_rate=${EMA_RATE} lr_warmup_steps=${WARMUP} \
  +run_name=${RUN} \
  2>&1 | tee logs/${RUN}.log
