#!/bin/bash
# Sudoku Figure 3 대조 셀 러너: GPU 샤딩 + 병합.
#
#   GPUS=4,5,6,7 STEP=step_65100 SEED=0 LIMIT=20000 N=20 HMS=16 \
#     OUT=results/fig3/ema_s0_n20_h16.json bash scripts/run_sudoku_fig3.sh
#
# 각 GPU 가 연속 행 블록 하나를 맡고, 끝나면 --merge 로 원 행 순서대로 합친다.
# ★ 샤딩 레이아웃이 바뀌면 행->노이즈 대응이 바뀐다 (RNG 는 배치 순서대로 소비).
#   평균은 표본오차 안에서 일치하지만 비트 동일하지는 않다.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CKPT="${CKPT:-checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34_origdata}"
STEP="${STEP:-step_65100}"
DATA="${DATA:-data/sudoku-extreme-1k-aug-1000/test}"
LIMIT="${LIMIT:-20000}"
OFFSET="${OFFSET:-0}"
N="${N:-20}"
HMS="${HMS:-16}"
SEED="${SEED:-0}"
SIGMA="${SIGMA:-1.0}"
BS="${BS:-512}"
KS="${KS:-1,5,20}"
OUT="${OUT:?OUT 필요}"

IFS=',' read -ra GARR <<< "$GPUS"
NS=${#GARR[@]}
# ★ 산출 디렉토리를 chaeun 으로 만든다.  root 셸이 mkdir 하면 755 root 소유가 되고
#   as-chaeun 쓰기가 마지막 줄에서 PermissionError 로 죽는다
#   (reports/INCIDENT_20260807_perm.md: nq8 σ 스윕 7셀 소실).
as-chaeun mkdir -p "$(dirname "$OUT")" logs
LOGBASE="logs/$(basename "${OUT%.json}")"

echo "[fig3] ckpt=$CKPT/$STEP data=$DATA rows=[$OFFSET,$((OFFSET+LIMIT))) N=$N hms=$HMS seed=$SEED shards=$NS gpus=$GPUS"

pids=()
for i in "${!GARR[@]}"; do
  as-chaeun env CUDA_VISIBLE_DEVICES="${GARR[$i]}" \
    python3 experiments/sudoku_fig3.py \
    --ckpt-dir "$CKPT" --step "$STEP" --data "$DATA" \
    --offset "$OFFSET" --limit "$LIMIT" --N "$N" --halt-max-steps "$HMS" \
    --seed "$SEED" --sigma-scale "$SIGMA" --batch-size "$BS" \
    --shard "$i" --num-shards "$NS" --out "$OUT" \
    > "${LOGBASE}.shard${i}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" != "0" ]; then
  echo "[fig3] FATAL: 샤드 실패. 로그: ${LOGBASE}.shard*.log"; exit 1
fi

as-chaeun python3 experiments/sudoku_fig3.py --merge "$OUT" --num-shards "$NS" --ks "$KS" \
  2>&1 | tee "${LOGBASE}.merge.log"
