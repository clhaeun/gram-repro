#!/bin/bash
# [50] Sudoku 채점 -- step_65100 EMA / N=20 / hms=16 / limit 20000 / hseed {0,1}
# 20,000행을 GPU 0-3 에 연속블록 4샤드로 나눈다.  샤드당 5,000행.
#   GPU=0 bash scripts/run_sudoku_50.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:?}"
CK=checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34_eval50
OUT=reports/sudoku50
mkdir -p "$OUT"
for HS in 0 1; do
  O=$OUT/n20000_hms16_N20_h${HS}_shard${GPU}of4.json
  [ -f "$O" ] && { echo "[skip] $O"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU python experiments/sudoku_harness.py \
    --ckpt-dir "$CK" --step step_65100 \
    --data data/sudoku-extreme-1k-aug-1000/test \
    --N 20 --halt-max-steps 16 --limit 20000 \
    --shard-index "$GPU" --shard-count 4 \
    --seed "$HS" --out "$O" \
    > "${O%.json}.log" 2>&1 || echo "!!! FAIL gpu$GPU hs$HS"
  echo "[done] hs$HS shard$GPU"
done
echo "[all done] gpu $GPU"
