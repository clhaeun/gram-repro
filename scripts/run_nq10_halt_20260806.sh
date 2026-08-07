#!/usr/bin/env bash
# nq10 A0 (nq10_repro_s0 / step_138534 / EMA / hseed 0) halt 스윕 보강.
#
#   halt 8   : 신규 -- 표 대칭용 (기존 halt 32/64 만 있어 아래쪽이 비어 있었다).
#   halt 32/64 : 재실행. 기존 results/a0_nq10_s0_final_ema_h0_halt{32,64}.json 은
#                queen_count 가 없다 (구 sweep_harness 가 산출 안 함).
#                동일 조건 재실행이라 acc/cov/maj/vsel 은 기존값 검산도 겸한다.
#   halt 16  : 학습값. 스윕 기준선이 harness 축 queen_count 로도 필요하다.
#
# nq10 test 는 19,220행이라 nq8(1,270행) 대비 대략 15배 걸린다.
# 사용: GPU=6 HMS="16 32" bash scripts/run_nq10_halt_20260806.sh
set -u

GPU="${GPU:?GPU=<id> 를 줘라}"
HMS="${HMS:-16 32 64 8}"   # 우선순위 순. halt 8 은 표 대칭용이라 맨 뒤.
OUT="results/halt_sweep_20260806"
CKDIR="checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0"
STEP="step_138534"          # 최종 = EMA 정점 (REPORT_final_pair_50 §2)

mkdir -p "$OUT" logs

for hms in $HMS; do
  out="${OUT}/nq10_repro_s0_final_ema_h0_hms${hms}.json"
  if [ -s "$out" ]; then echo "SKIP (이미 있음) $out"; continue; fi
  echo "=== RUN nq10 repro_s0 EMA hms=${hms}  $(date -u +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$GPU" python3 experiments/sweep_harness.py \
    --ckpt-dir "$CKDIR" --step "$STEP" \
    --data data/nqueens-10/test --n 10 --N 20 --seed 0 \
    --sidecar data/nqueens-10/sidecar_10.json \
    --halt-max-steps "$hms" --out "$out" \
    >> logs/halt_sweep_nq10_s0.log 2>&1 \
    || echo "!!! FAIL nq10 hms=${hms}"
done

echo "=== DONE nq10 repro_s0"
