#!/bin/bash
# [50] Fig 4 right / Fig 15 per-puzzle 덤프.  N 마다 독립 프로세스.
# EMA(bare step 파일) / sigma_scale 1.0 / mu_scale 1.0 / hms 16 / harness seed 0 / test 전수.
set -u
cd /workspace/TinyRecursiveModels
LOG=logs/perpuzzle_50.log
: > $LOG

dump() {  # $1 gpu  $2 tag  $3 ckptdir  $4 step  $5 data  $6 sidecar  $7 n  $8 N
  local O=results/figdata/$2_perpuzzle_N$8.csv
  [ -f "$O" ] && { echo "[skip] $O" >> $LOG; return; }
  CUDA_VISIBLE_DEVICES=$1 python experiments/perpuzzle_dump.py \
    --ckpt-dir "$3" --step "$4" --data "$5" --sidecar "$6" --n "$7" \
    --N "$8" --seed 0 --sigma-scale 1.0 --mu-scale 1.0 --halt-max-steps 16 \
    --weight ema --out "$O" >> $LOG 2>&1 \
    || echo "!!! FAIL $2 N=$8" >> $LOG
}

NQ8S0=checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0
NQ8S1=checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s1
NQ10=checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0
D8="data/nqueens-8/test data/nqueens-8/sidecar_8.json 8"
D10="data/nqueens-10/test data/nqueens-10/sidecar_10.json 10"

# GPU 큐 (큐 안은 순차, 큐끼리는 병렬)
( dump 0 nq8 $NQ8S0 step_67428 $D8 5 ; dump 0 nq8 $NQ8S0 step_67428 $D8 20 ) &
( dump 1 nq8 $NQ8S0 step_67428 $D8 10 ; dump 1 nq8_s1 $NQ8S1 step_67428 $D8 20 ) &
( dump 2 nq8_s1 $NQ8S1 step_67428 $D8 1 ; dump 2 nq8_s1 $NQ8S1 step_67428 $D8 5 ; \
  dump 2 nq8_s1 $NQ8S1 step_67428 $D8 10 ) &
( dump 4 nq10 $NQ10 step_138534 $D10 1 ) &
( dump 5 nq10 $NQ10 step_138534 $D10 5 ) &
( dump 6 nq10 $NQ10 step_138534 $D10 10 ) &
( dump 7 nq10 $NQ10 step_138534 $D10 20 ) &
wait
echo "ALL DONE $(date +%m%d-%H:%M)" >> $LOG
