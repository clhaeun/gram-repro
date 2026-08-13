#!/bin/bash
# [50] GRAM ε 전파 probe 러너 — task 하나를 GPU 하나에 올려 sigma -> lambda 순으로 돈다.
# 추론 전용.  재학습 없음.
#
#   GPU=4 TASK=nq10 bash experiments/run_eps_probe_50.sh
#
# ★ root 로 돌리지 않는다 (reports/INCIDENT_20260807_perm.md).
#   이 스크립트는 uid 1048 을 요구하고, 아니면 즉시 죽는다.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(id -u)" != "1048" ]; then
  echo "[eps-probe] uid=$(id -u) — chaeun(1048) 으로만 실행한다. 중단." >&2
  exit 1
fi

GPU="${GPU:?}"; TASK="${TASK:?}"
N="${N:-256}"; BS="${BS:-64}"; RATIOS="${RATIOS:-0.01,0.1,1.0}"

case "$TASK" in
  nq10)   CKPT=checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0;                       STEP=step_138534; DATA=data/nqueens-10/test ;;
  nq8)    CKPT=checkpoints/Nqueens-8-ACT-torch/nq8_A0_e3000_s0;                      STEP=step_28095;  DATA=data/nqueens-8/test ;;
  gc8)    CKPT=checkpoints/Graph-coloring-8-ACT-torch/gc8_repro_s0;                  STEP=step_9400;   DATA=data/graph-coloring-8/test ;;
  # gc8_repro_s0 은 붕괴 셀(EMA exact_acc/kl_loss 0.0000)이다.  coverage 0.82 셀로
  # 다시 재기 위한 분기 — checkpoint 는 34번에서 받는다
  # (reports/TRANSFER_34_to_50_gc8paper.md).
  gc8p)   CKPT=checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34;                  STEP=step_45540;  DATA=data/graph-coloring-8/test ;;
  # gc10 최종 세팅.  checkpoint 는 34번에서 받는다
  # (reports/TRANSFER_34_to_50_gc10paper.md).  test 는 192행뿐이라 N=192 로 돈다.
  gc10p)  CKPT=checkpoints/Graph-coloring-10-ACT-torch/gc10_paper_34;                STEP=step_87620;  DATA=data/graph-coloring-10/test ;;
  # nq8 기준 run.  nq8_A0_e3000_s0 과는 epochs 3000->7200 및 y_* / kl_reduction
  # 키가 다르다 (아래 보고서 ④ 참조).
  nq8fa)  CKPT=checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0;                     STEP=step_67428;  DATA=data/nqueens-8/test ;;
  sudoku) CKPT=checkpoints/Sudoku-extreme-1k-aug-1000-test10k-ACT-torch/gram_sudoku_paper; STEP=step_65100; DATA=data/sudoku-extreme-1k-aug-1000-test10k/test ;;
  # sudoku 최종 세팅 (sigma_init 1.0).  test10k-v2 는 이 서버에 없었으므로
  # exp/make_test10k_v2.py (seed 42 고정) 로 로컬 생성했다.
  sudoku34) CKPT=checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34; STEP=step_65100; DATA=data/sudoku-extreme-1k-aug-1000-test10k-v2/test ;;
  *) echo "unknown TASK=$TASK" >&2; exit 1 ;;
esac

OUT=results/eps_probe
mkdir -p "$OUT" logs

export CUDA_VISIBLE_DEVICES="$GPU"
export HOME=/workspace/home/chaeun
export TRITON_CACHE_DIR=/workspace/home/chaeun/.triton
export XDG_CACHE_HOME=/workspace/home/chaeun/.cache
export OMP_NUM_THREADS=4

echo "[eps-probe] task=$TASK gpu=$GPU ckpt=$CKPT step=$STEP n=$N uid=$(id -u)"

python3 experiments/eps_propagation_probe.py --mode sigma \
  --ckpt-dir "$CKPT" --step "$STEP" --data "$DATA" \
  --n-puzzles "$N" --batch-size "$BS" --posterior \
  --out "$OUT/${TASK}_sigma.json"
echo "[eps-probe] sigma exit=$?"

python3 experiments/eps_propagation_probe.py --mode lambda \
  --ckpt-dir "$CKPT" --step "$STEP" --data "$DATA" \
  --n-puzzles "$N" --batch-size "$BS" --ratios "$RATIOS" \
  --out "$OUT/${TASK}_lambda.json"
echo "[eps-probe] lambda exit=$? task=$TASK"
