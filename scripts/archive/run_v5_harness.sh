#!/bin/bash
# v5 4-run harness. 정점 step + 최종 step(68340) x {EMA, .live} x harness seed {0,1} = 32 건.
#   GPU 0 = 우선순위 6 건(add_b0 3seed x 68340 x EMA x seed{0,1})을 먼저 돌린 뒤 나머지 짝수 index.
#   GPU 1 = 나머지 홀수 index.
# 출력: experiments/results/v5h_<run>_step<STEP>_<ema|live>_s<seed>.json
#   ★ 파일명에 step 이 들어간다. band_harness.sh 에서 step 누락으로 정점 결과가
#     최종 결과에 덮여 날아간 적이 있다.  같은 실수 반복 금지.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # scripts/archive/ -> repo 루트
source ~/miniconda3/bin/activate gram

WHICH="${1:?usage: run_v5_harness.sh <gpu0|gpu1>}"
CK=checkpoints/Nqueens-8-ACT-torch
OUT=experiments/results

# run:정점step
PEAK=(
  "nq8_v5_add_b0_b500_s0:54672"
  "nq8_v5_add_b0_b500_s1:43282"
  "nq8_v5_add_b0_b500_s2:66062"
  "nq8_v5_add_bm4_b500_s1:53533"
)
FINAL=68340

JOBS=()
# 우선순위: add_b0 3seed x 최종 x EMA x harness seed 0/1
for R in nq8_v5_add_b0_b500_s0 nq8_v5_add_b0_b500_s1 nq8_v5_add_b0_b500_s2; do
  for HS in 0 1; do JOBS+=("$R:$FINAL:ema:$HS"); done
done
# 나머지
for E in "${PEAK[@]}"; do
  R="${E%%:*}"; P="${E##*:}"
  for HS in 0 1; do
    JOBS+=("$R:$P:ema:$HS")
    JOBS+=("$R:$P:live:$HS")
    JOBS+=("$R:$FINAL:live:$HS")
  done
done
for HS in 0 1; do JOBS+=("nq8_v5_add_bm4_b500_s1:$FINAL:ema:$HS"); done

# 분배: 앞 6 건(우선순위)은 gpu0 이 독점.  나머지 26 건은 gpu0 10 / gpu1 16 으로 나눠
# 양쪽 다 16 건이 되게 한다.  job 당 비용(N=20 x 772 퍼즐)은 전부 같다.
MINE=()
for i in "${!JOBS[@]}"; do
  if [ "$i" -lt 16 ]; then
    [ "$WHICH" = "gpu0" ] && MINE+=("${JOBS[$i]}")
  else
    [ "$WHICH" = "gpu1" ] && MINE+=("${JOBS[$i]}")
  fi
done

GPU=$([ "$WHICH" = "gpu0" ] && echo 0 || echo 1)
echo "[$WHICH] GPU=$GPU  총 ${#MINE[@]} 건 / 전체 ${#JOBS[@]} 건"
printf '  %s\n' "${MINE[@]}"

for J in "${MINE[@]}"; do
  IFS=':' read -r R S W HS <<< "$J"
  [ "$W" = "ema" ] && SF="step_$S" || SF="step_$S.live"
  O="$OUT/v5h_${R}_step${S}_${W}_s${HS}.json"
  if [ -f "$O" ]; then echo "[skip] $O 이미 존재"; continue; fi
  echo "=== $(date +%H:%M:%S) $R $SF harness_seed=$HS -> $O"
  CUDA_VISIBLE_DEVICES=$GPU python experiments/sweep_harness.py \
    --ckpt-dir "$CK/$R" --step "$SF" \
    --data data/nqueens-8/test --sidecar data/nqueens-8/sidecar_8.json --n 8 \
    --N 20 --seed "$HS" --tag "v5h_${R}_${S}_${W}" --out "$O" \
    || echo "!!! FAIL $J"
done
echo "[$WHICH] 완료 $(date +%H:%M:%S)"
