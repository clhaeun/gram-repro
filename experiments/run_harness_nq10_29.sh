#!/usr/bin/env bash
# 29번 nq10 최종 harness (crash 로 125940 step 에서 조기 종료된 2 run).
#   {최종 125940, EMA 정점, raw 정점} x {EMA(step_N), live(step_N.live)} x hseed 0/1
#   = run 당 12건, 총 24건.  GPU 0~7 에 3건씩 분배.
# 정점 step 출처: logs/<run>.log 의 [EVAL @ step](=EMA) / [EVAL RAW @ step](=live)
#   nq10_repro_s2   EMA 정점 104950 (0.9507) / raw 정점  88158 (0.9359)
#   nq10_fp_a3p_s1  EMA 정점 121742 (0.5791) / raw 정점 113346 (0.7882)
set -u
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/gram/bin/python
CK=checkpoints/Nqueens-10-ACT-torch

JOBS=()
for spec in "nq10_repro_s2 125940 104950 88158" "nq10_fp_a3p_s1 125940 121742 113346"; do
  set -- $spec; R=$1; shift
  for S in "$@"; do
    for W in ema live; do
      [ "$W" = ema ] && SF="step_${S}" || SF="step_${S}.live"
      for HS in 0 1; do
        JOBS+=("$R|$SF|$HS|fin_${R}_step${S}_${W}_s${HS}")
      done
    done
  done
done

echo "총 ${#JOBS[@]} 건"
if [ "${DRYRUN:-0}" = 1 ]; then printf '%s\n' "${JOBS[@]}"; exit 0; fi

run_worker() {
  local g=$1
  for ((i=g; i<${#JOBS[@]}; i+=8)); do
    IFS='|' read -r R SF HS TAG <<<"${JOBS[$i]}"
    O="experiments/results/${TAG}.json"
    if [ -f "$O" ]; then echo "[gpu$g] SKIP exists $TAG"; continue; fi
    echo "[gpu$g] $(date '+%H:%M:%S') START $TAG"
    CUDA_VISIBLE_DEVICES=$g $PY experiments/sweep_harness.py \
      --ckpt-dir "$CK/$R" --step "$SF" \
      --data data/nqueens-10/test --sidecar data/nqueens-10/sidecar_10.json --n 10 \
      --N 20 --seed "$HS" --out "$O"
    echo "[gpu$g] $(date '+%H:%M:%S') DONE  $TAG rc=$?"
  done
}

for g in 0 1 2 3 4 5 6 7; do
  run_worker $g > "logs/harness_nq10_gpu${g}.log" 2>&1 &
done
wait
echo "ALL DONE $(date '+%F %T')"
