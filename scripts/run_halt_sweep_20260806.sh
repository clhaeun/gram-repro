#!/usr/bin/env bash
# halt_max_steps 스윕 + nq8 최종/정점 전 지표 재측정 (사수 지시 2026-08-06).
#
# 목적 3가지
#   1. nq8 최종(nq8_final_add_s0/s1) 전 지표 -- 특히 기존 fin_*.json 에 없는
#      cov MACRO(퍼즐 축). 구 harness 는 per_row 를 안 남겨 재집계가 불가능했다.
#   2. nq8 halt 스윕 8/16/32/64 (N_sup 외삽). test 1,270행이라 nq10 보다 빠르다.
#   3. queen_count -- sweep_harness.py 가 이번에 산출하기 시작한 축
#      (전샘플 K*B. 학습 로그의 in-loop 단일샘플 축과 눈금 다름).
#
# 축 규약: hseed 0 고정 (기존 실측에서 hseed 편차 최대 0.5pp). N=20, (mu,sigma)=(1,1).
# 사용: SEED=s0 GPU=6 bash scripts/run_halt_sweep_20260806.sh
set -u

SEED="${SEED:?SEED=s0|s1 를 줘라}"
GPU="${GPU:?GPU=<id> 를 줘라}"
OUT="results/halt_sweep_20260806"
HARNESS="experiments/sweep_harness.py"
CKDIR="checkpoints/Nqueens-8-ACT-torch/nq8_final_add_${SEED}"

# 최종 step 은 s0/s1 공통 67,428.
# ★ nq8 은 in-loop EMA strict 정점이 s0/s1 둘 다 67,428 = 최종이다 (logs/nq8_final_add_s*.log
#   실측: s0 67428=0.9992 > 65555=0.9984, s1 67428=0.9882 > 65555=0.9843).
#   즉 "최종 vs 정점" 이 갈리는 nq10 s1 같은 상황이 nq8 엔 없다.
#   아래 MID 는 정점이 아니라, 구 harness 가 이미 재본 중간 step 이다 (축 대조용 병기).
FINAL=67428
case "$SEED" in
  s0) MID=29968 ;;
  s1) MID=59936 ;;
  *)  echo "SEED 는 s0|s1"; exit 1 ;;
esac

mkdir -p "$OUT" logs

run () {   # $1=step  $2=ema|live  $3=halt  $4=태그
  local step="$1" w="$2" hms="$3" tag="$4"
  local ck="step_${step}"; [ "$w" = live ] && ck="step_${step}.live"
  local out="${OUT}/nq8_${SEED}_${tag}_${w}_h0_hms${hms}.json"
  if [ -s "$out" ]; then echo "SKIP (이미 있음) $out"; return; fi
  echo "=== RUN nq8 ${SEED} ${tag}(${step}) ${w} hms=${hms}"
  CUDA_VISIBLE_DEVICES="$GPU" python3 "$HARNESS" \
    --ckpt-dir "$CKDIR" --step "$ck" \
    --data data/nqueens-8/test --n 8 --N 20 --seed 0 \
    --sidecar data/nqueens-8/sidecar_8.json \
    --halt-max-steps "$hms" --out "$out" \
    >> "logs/halt_sweep_nq8_${SEED}.log" 2>&1 \
    || echo "!!! FAIL nq8 ${SEED} ${tag} ${w} hms=${hms}"
}

# (2) halt 스윕 -- 최종 step, EMA/live 양축. 짧은 것부터 (조기 신호 확보).
for hms in 8 16 32 64; do
  for w in ema live; do
    run "$FINAL" "$w" "$hms" final
  done
done

# (1) 중간 step 병기 -- 학습 halt 값(16)에서만. 최종 halt16 은 위 루프가 이미 낸다.
for w in ema live; do
  run "$MID" "$w" 16 mid
done

echo "=== DONE nq8 ${SEED}"
