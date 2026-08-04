#!/bin/bash
# nq10_repro harness: {최종, EMA 정점, raw 정점} x {EMA, .live} x hseed{0,1}.
# 정점은 in-loop valid 곡선에서 선정 (s0: EMA정점=최종, raw정점 134336 /
# s1: raw정점=최종, EMA정점 67168).  GPU 하나당 한 (run, step) 묶음 4건.
#   GPU=4 RUN=nq10_repro_s0 STEP=138534 bash experiments/run_repro_harness.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:?}"; RUN="${RUN:?}"; STEP="${STEP:?}"
CK=checkpoints/Nqueens-10-ACT-torch/$RUN
for W in ema live; do
  [ "$W" = ema ] && SF=step_$STEP || SF=step_$STEP.live
  for HS in 0 1; do
    O=experiments/results/rp_${RUN}_step${STEP}_${W}_s${HS}.json
    [ -f "$O" ] && { echo "[skip] $O"; continue; }
    CUDA_VISIBLE_DEVICES=$GPU ~/venv-gram/bin/python experiments/sweep_harness.py \
      --ckpt-dir "$CK" --step "$SF" \
      --data data/nqueens-10/test --sidecar data/nqueens-10/sidecar_10.json \
      --n 10 --N 20 --seed "$HS" --out "$O" || echo "!!! FAIL $RUN $STEP $W hs$HS"
  done
done
echo "[done] $RUN step $STEP"
