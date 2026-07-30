#!/bin/bash
# 50번 β sweep (one-hot ON) harness. 인자: <run> <step> <gpu>
# seed 는 prior 샘플링 RNG (학습 편차 아님).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=$1; shift; GPU=$1; shift
for STEP in "$@"; do
  for SEED in 0 1; do
    OUT="experiments/results/sweep50_${RUN}_step_${STEP}_s${SEED}.json"
    [ -e "checkpoints/Nqueens-8-ACT-torch/${RUN}/step_${STEP}" ] || { echo "[skip] ${RUN} step_${STEP} 없음"; continue; }
    echo "=== ${RUN} step_${STEP} seed=${SEED} (GPU ${GPU}) ==="
    CUDA_VISIBLE_DEVICES=${GPU} python experiments/sweep_harness.py \
      --ckpt-dir "checkpoints/Nqueens-8-ACT-torch/${RUN}" --step "step_${STEP}" \
      --seed "${SEED}" --tag "${RUN} step_${STEP} seed${SEED}" --out "${OUT}"
  done
done
echo "[done] ${RUN}"
