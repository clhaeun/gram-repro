#!/bin/bash
# ############################################################################
# DEPRECATED -- scripts/run_repro_harness.sh 를 쓸 것.
#
# 이 스크립트는 현행 코드로 실행 불가다 (고치지 않고 표시만 남긴다):
#   1. `source ~/miniconda3/bin/activate gram`  -- 34번에 ~/miniconda3 가 없다.
#      현행 환경은 ~/venv-gram (scripts/run_repro_harness.sh 는 ~/venv-gram/bin/python 사용).
#   2. `python experiments/sweep_harness.py`    -- `python` 실행파일이 PATH 에 없다.
#      절대경로 ~/venv-gram/bin/python 을 써야 한다.
#   3. `--tag "finh_..."`                       -- sweep_harness.py 에 --tag 인자가 없다.
#      현행 인자: --ckpt-dir --step --data --n --N --seed --sigma-scale --mu-scale
#                 --sidecar --halt-max-steps --out.  argparse 가 즉시 죽는다.
# ############################################################################
# nq8_final_* harness 28 건.  GPU 3장(5/6/7)에 shard 로 나눠 돈다.
#
#   catnd_s0 : 67428(최종) / 50571(EMA정점) / 44952(raw정점)  x {EMA,.live} x hseed{0,1} = 12
#   catnd_s1 : 67428(최종) / 58063(EMA정점) / 63682(raw정점)  x {EMA,.live} x hseed{0,1} = 12
#   add_s2   : 67428 만 (정점 == 최종)                        x {EMA,.live} x hseed{0,1} =  4
#
# ★ 데이터는 새 pair 데이터다 (data/nqueens-8/test, 1,270행 / 772 unique 입력).
#   -uniq 가 아니다.  모델이 pair 데이터로 학습됐으므로 평가도 pair test 로 한다.
# ★ 출력 파일명에 step 이 들어간다.  전에 step 을 빼먹어 정점 결과가 최종 결과에
#   덮여 날아간 적이 있다.  같은 실수 반복 금지.
# ★ launch 전 GPU 점유 확인(>1000MiB 면 중단).  다중 사용자다.
#
# 사용법 (GPU 하나당 한 줄, 각각 다른 tmux 창):
#   GPU=5 SHARD=0 NSHARD=3 bash experiments/run_final_harness.sh
#   GPU=6 SHARD=1 NSHARD=3 bash experiments/run_final_harness.sh
#   GPU=7 SHARD=2 NSHARD=3 bash experiments/run_final_harness.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ~/miniconda3/bin/activate gram

GPU="${GPU:?GPU=<idx> 를 줘라}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
CK=checkpoints/Nqueens-8-ACT-torch
OUT="${OUT:-experiments/results}"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
[ "$used" -gt 1000 ] && { echo "FATAL: GPU $GPU 사용 중 (${used}MiB) -- skip"; exit 1; }

# 우선순위: 헤드라인(최종 step x EMA) 3 run x hseed{0,1} = 6 건을 맨 앞에.
JOBS=()
for R in nq8_final_add_s2 nq8_final_catnd_s0 nq8_final_catnd_s1; do
  for HS in 0 1; do JOBS+=("$R:67428:ema:$HS"); done
done
for R in nq8_final_add_s2 nq8_final_catnd_s0 nq8_final_catnd_s1; do
  for HS in 0 1; do JOBS+=("$R:67428:live:$HS"); done
done
for E in "nq8_final_catnd_s0:50571" "nq8_final_catnd_s0:44952" \
         "nq8_final_catnd_s1:58063" "nq8_final_catnd_s1:63682"; do
  R="${E%%:*}"; S="${E##*:}"
  for W in ema live; do for HS in 0 1; do JOBS+=("$R:$S:$W:$HS"); done; done
done

echo "[shard $SHARD/$NSHARD] GPU=$GPU  전체 ${#JOBS[@]} 건"
mkdir -p "$OUT"

for i in "${!JOBS[@]}"; do
  [ $(( i % NSHARD )) -eq "$SHARD" ] || continue
  IFS=':' read -r R S W HS <<< "${JOBS[$i]}"
  [ "$W" = "ema" ] && SF="step_$S" || SF="step_$S.live"
  O="$OUT/finh_${R}_step${S}_${W}_s${HS}.json"
  if [ -f "$O" ]; then echo "[skip] $O 이미 존재"; continue; fi
  echo "=== $(date +%H:%M:%S) $R $SF hseed=$HS -> $O"
  CUDA_VISIBLE_DEVICES=$GPU python experiments/sweep_harness.py \
    --ckpt-dir "$CK/$R" --step "$SF" \
    --data data/nqueens-8/test --sidecar data/nqueens-8/sidecar_8.json --n 8 \
    --N 20 --seed "$HS" --tag "finh_${R}_${S}_${W}" --out "$O" \
    || echo "!!! FAIL ${JOBS[$i]}"
done
echo "[shard $SHARD] 완료 $(date +%H:%M:%S)"
