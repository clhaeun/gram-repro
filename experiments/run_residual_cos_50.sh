#!/bin/bash
# [50 ③] 잔차 프로브에 cos(eps_t, u_t - h_(t-1)) 추가하고 GRAM sigma 셀 재실행.
# FPRM 셀은 대상 아님.  셀당 forward 1회 (n_probe 256 + n_traj 50).
set -u
cd /workspace/TinyRecursiveModels
LOG=logs/residual_cos_50.log
: > $LOG

run() {  # gpu key ckptdir step data sigma
  echo "[start gpu$1] $2 sigma=$6 $(date +%H:%M)" >> $LOG
  CUDA_VISIBLE_DEVICES=$1 python experiments/residual_probe.py \
    --key "$2" --ckpt-dir "$3" --step "$4" --data "$5" --sigma "$6" \
    --n-probe 256 --n-traj 50 --traj-index 0 --seed 0 >> $LOG 2>&1 \
    || echo "!!! FAIL $2" >> $LOG
}

NQ10=checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0
NQ8=checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0
GC8=checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34
SUD=checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34

( run 0 nq10_sig0 $NQ10 step_138534 data/nqueens-10/test 0.0
  run 0 nq10_sig1 $NQ10 step_138534 data/nqueens-10/test 1.0 ) &
( run 1 nq8_sig0  $NQ8  step_67428  data/nqueens-8/test  0.0
  run 1 nq8_sig1  $NQ8  step_67428  data/nqueens-8/test  1.0 ) &
( run 2 gc8_sig0  $GC8  step_45540  data/graph-coloring-8/test 0.0
  run 2 gc8_sig1  $GC8  step_45540  data/graph-coloring-8/test 1.0 ) &
( run 3 sudoku_sig0 $SUD step_65100 data/sudoku-extreme-1k-aug-1000-test10k-v2/test 0.0
  run 3 sudoku_sig1 $SUD step_65100 data/sudoku-extreme-1k-aug-1000-test10k-v2/test 1.0 ) &
wait
echo "RESIDUAL COS DONE $(date +%m%d-%H:%M)" >> $LOG
