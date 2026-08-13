#!/bin/bash
# [50] ⑥ depth utilization -- FPRM 4.4.2 (post-norm 이 pre-norm 보다 일찍 포화) 를 GRAM 에 이식.
#
# 62번 격자(hms 8~64)와 표본·조건을 그대로 맞추고 128, 320 만추가한다.
#   ckpt   sudoku_paper_34 / step_65100 (EMA; step_65100.live 가 live)
#          ※ 로컬 hms16 기준선은 sudoku_paper_34_eval50 로 돌았지만 step 파일은
#            md5 b050e90165c998abf79f55b3e27193c6 로 바이트 동일하다.
#   data   data/sudoku-extreme-1k-aug-1000/test  앞 20,000행 (limit 20000)
#   N=20   sigma=1.0  mu=1.0  seed(h)=0
# 20,000행을 GPU 0-7 에 연속블록 8샤드(샤드당 2,500행)로 나눈다.  경계는 harness 의
# np.linspace 라 4샤드 기준선과 합집합이 정확히 같다.
#
#   as-chaeun bash scripts/run_depth_util_50.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CK=checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34
OUT=reports/sudoku50
HSEED="${HSEED:-0}"
mkdir -p "$OUT"

for HS in 128 320; do
  echo "== hms=$HS 시작 $(date -u +%H:%M:%S)"
  for g in 0 1 2 3 4 5 6 7; do
    O=$OUT/n20000_hms${HS}_N20_h${HSEED}_shard${g}of8.json
    if [ -f "$O" ]; then echo "[skip] $O"; continue; fi
    CUDA_VISIBLE_DEVICES=$g python experiments/sudoku_harness.py \
      --ckpt-dir "$CK" --step step_65100 \
      --data data/sudoku-extreme-1k-aug-1000/test \
      --N 20 --halt-max-steps "$HS" --limit 20000 \
      --shard-index "$g" --shard-count 8 \
      --seed "$HSEED" --out "$O" \
      > "${O%.json}.log" 2>&1 || echo "!!! FAIL hms$HS gpu$g" &
  done
  wait
  M=$OUT/n20000_hms${HS}_N20_h${HSEED}_merged8.json
  python experiments/merge_sudoku_shards.py "$M" \
    $OUT/n20000_hms${HS}_N20_h${HSEED}_shard*of8.json && echo "[merged] $M"
  echo "== hms=$HS 완료 $(date -u +%H:%M:%S)"
done

python experiments/depth_util_summary.py
