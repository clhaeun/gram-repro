#!/bin/bash
# [50 ②-a] queen_key 정규화 후 GRAM per-puzzle 덤프 재집계.
# 기존 CSV 는 /tmp/.../precanon 에 백업돼 있고 여기서는 덮어쓴다.
set -u
cd /workspace/TinyRecursiveModels
LOG=logs/perpuzzle_canon.log
: > $LOG

dump() {  # $1 gpu  $2 tag  $3 ckptdir  $4 step  $5 data  $6 sidecar  $7 n  $8 N
  CUDA_VISIBLE_DEVICES=$1 python experiments/perpuzzle_dump.py \
    --ckpt-dir "$3" --step "$4" --data "$5" --sidecar "$6" --n "$7" \
    --N "$8" --seed 0 --sigma-scale 1.0 --mu-scale 1.0 --halt-max-steps 16 \
    --weight ema --out "results/figdata/$2_perpuzzle_N$8.csv" >> $LOG 2>&1 \
    || echo "!!! FAIL $2 N=$8" >> $LOG
}

NQ8S0=checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0
NQ8S1=checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s1
NQ10=checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0
D8="data/nqueens-8/test data/nqueens-8/sidecar_8.json 8"
D10="data/nqueens-10/test data/nqueens-10/sidecar_10.json 10"

( for N in 1 5 10 20; do dump 0 nq8    $NQ8S0 step_67428  $D8  $N; done ) &
( for N in 1 5 10 20; do dump 1 nq8_s1 $NQ8S1 step_67428  $D8  $N; done ) &
( dump 2 nq10 $NQ10 step_138534 $D10 1 ; dump 2 nq10 $NQ10 step_138534 $D10 5 ) &
( dump 3 nq10 $NQ10 step_138534 $D10 10 ) &
( dump 4 nq10 $NQ10 step_138534 $D10 20 ) &
wait
echo "CANON DONE $(date +%m%d-%H:%M)" >> $LOG
