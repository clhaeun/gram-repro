#!/bin/bash
# v6 harness 24 건.  29번에서 학습한 run 을 50번에서 재도록 만든 러너다.
#
#   8x8  nq8_v6_add_b0_noOH_s{0,1,2}  step_68340   x {EMA, .live} x harness seed {0,1} = 12
#   10x10 nq10_v6_add_b0_s{0,1,2}     step_60522   x {EMA, .live} x harness seed {0,1} = 12
#
# 목적: y_onehot ON/OFF 판정이 지금 in-loop(퍼즐당 1샘플) 기준이라 v5(harness, 20샘플)와
# 축이 안 맞는다.  같은 축으로 맞추려면 이 24 건이 필요하다.
#
# 사용법 (50번에서, GPU 하나당 한 줄):
#   GPU=1 SHARD=0 NSHARD=5 bash experiments/run_v6_harness.sh
#   GPU=2 SHARD=1 NSHARD=5 bash experiments/run_v6_harness.sh
#   ...
#
# 체크포인트/데이터 위치는 환경변수로 바꾼다.  기본값은 29번이 스테이징한 공유 NFS 경로다.
#   CKPT_ROOT=/data/chaeun/gram-share/ckpt
#   DATA_ROOT=/data/chaeun/gram-share/data
# 50번이 /data 를 마운트하지 않으면 로컬로 복사한 뒤 두 변수를 그 경로로 주면 된다.
# 무결성은 /data/chaeun/gram-share/MANIFEST.md5 로 대조할 수 있다.
#
# ★ launch 전 반드시 nvidia-smi 로 해당 GPU 점유 확인 (>1000MiB 면 skip).  다중 사용자다.
# ★ 출력 파일명에 step 이 들어간다.  전에 step 을 빼먹어 정점 결과가 최종 결과에
#   덮여 날아간 적이 있다.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU="${GPU:?GPU=<idx> 를 줘라}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
CKPT_ROOT="${CKPT_ROOT:-/data/chaeun/gram-share/ckpt}"
DATA_ROOT="${DATA_ROOT:-/data/chaeun/gram-share/data}"
OUT="${OUT:-experiments/results}"
NSAMP="${NSAMP:-20}"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
[ "$used" -gt 1000 ] && { echo "FATAL: GPU $GPU 사용 중 (${used}MiB) -- skip"; exit 1; }

JOBS=()
for R in nq8_v6_add_b0_noOH_s0 nq8_v6_add_b0_noOH_s1 nq8_v6_add_b0_noOH_s2; do
  for W in ema live; do for HS in 0 1; do JOBS+=("8:Nqueens-8-ACT-torch:$R:68340:$W:$HS"); done; done
done
for R in nq10_v6_add_b0_s0 nq10_v6_add_b0_s1 nq10_v6_add_b0_s2; do
  for W in ema live; do for HS in 0 1; do JOBS+=("10:Nqueens-10-ACT-torch:$R:60522:$W:$HS"); done; done
done

echo "[shard $SHARD/$NSHARD] GPU=$GPU  전체 ${#JOBS[@]} 건"
mkdir -p "$OUT"

for i in "${!JOBS[@]}"; do
  [ $(( i % NSHARD )) -eq "$SHARD" ] || continue
  IFS=':' read -r N PROJ R S W HS <<< "${JOBS[$i]}"
  [ "$W" = "ema" ] && SF="step_$S" || SF="step_$S.live"
  O="$OUT/v6h_${R}_step${S}_${W}_s${HS}.json"
  if [ -f "$O" ]; then echo "[skip] $O"; continue; fi
  echo "=== $(date +%H:%M:%S) $R $SF hseed=$HS -> $O"
  CUDA_VISIBLE_DEVICES=$GPU python experiments/sweep_harness.py \
    --ckpt-dir "$CKPT_ROOT/$PROJ/$R" --step "$SF" \
    --data "$DATA_ROOT/nqueens-$N/test" --sidecar "$DATA_ROOT/nqueens-$N/sidecar_$N.json" \
    --n "$N" --N "$NSAMP" --seed "$HS" --tag "v6h_${R}_${S}_${W}" --out "$O" \
    || echo "!!! FAIL ${JOBS[$i]}"
done
echo "[shard $SHARD] 완료 $(date +%H:%M:%S)"
