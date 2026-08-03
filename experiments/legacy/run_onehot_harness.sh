#!/bin/bash
# one-hot 3-run harness: 각 run 의 peak + final checkpoint 를, harness seed 2개로.
#
# seed 는 sweep_harness.py 의 --seed = **prior 샘플링 RNG** 다. 즉 여기서 얻는 편차는
# "같은 checkpoint 를 다시 샘플했을 때의 편차"이지, 학습 run-to-run 비결정성이 아니다.
# 학습 편차를 보려면 학습을 다시 돌려야 한다. 혼동 금지.
#
# 사용법: bash experiments/run_onehot_harness.sh <run_name> <peak_step> <gpu>
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN=$1; PEAK=$2; GPU=$3
CKPT="checkpoints/Nqueens-8-ACT-torch/${RUN}"
FINAL=17085

for STEP in ${PEAK} ${FINAL}; do
  for SEED in 0 1; do
    OUT="experiments/results/onehot_${RUN}_step_${STEP}_s${SEED}.json"
    if [ ! -e "${CKPT}/step_${STEP}" ]; then   # checkpoint 는 디렉터리가 아니라 파일이다
      echo "[skip] ${CKPT}/step_${STEP} 없음"; continue
    fi
    echo "=== ${RUN} step_${STEP} seed=${SEED} (GPU ${GPU}) ==="
    CUDA_VISIBLE_DEVICES=${GPU} python experiments/sweep_harness.py \
      --ckpt-dir "${CKPT}" --step "step_${STEP}" --seed "${SEED}" \
      --tag "${RUN} step_${STEP} seed${SEED}" --out "${OUT}"
  done
done
echo "[done] ${RUN}"
