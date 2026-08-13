#!/bin/bash
# [50] ②~⑤ residual probe + 궤적 덤프.  10 셀을 GPU 0-7 에 배분한다.
#   as-chaeun bash scripts/run_residual_probe_50.sh
#
# GRAM 8 셀 (4 모델 x sigma {1.0, 0.0}) -> GPU 0-7 동시.
# FPRM 2 셀 -> GRAM 이 끝난 뒤 GPU 0-1.
# 산출은 전부 results/figdata (= /workspace/fprm/results/figdata 심볼릭 링크).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=reports/residual50
mkdir -p "$LOG"

TRM=/workspace/TinyRecursiveModels
FPRM=/workspace/fprm

# key|ckpt-dir|step|data|sigma|n_traj
GRAM_CELLS=(
  "nq10_sig1|checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0|step_138534|data/nqueens-10/test|1.0|50"
  "nq10_sig0|checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0|step_138534|data/nqueens-10/test|0.0|1"
  "nq8_sig1|checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0|step_67428|data/nqueens-8/test|1.0|50"
  "nq8_sig0|checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0|step_67428|data/nqueens-8/test|0.0|1"
  "gc8_sig1|checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34|step_45540|data/graph-coloring-8/test|1.0|50"
  "gc8_sig0|checkpoints/Graph-coloring-8-ACT-torch/gc8_paper_34|step_45540|data/graph-coloring-8/test|0.0|1"
  "sudoku_sig1|checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34|step_65100|data/sudoku-extreme-1k-aug-1000-test10k-v2/test|1.0|50"
  "sudoku_sig0|checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34|step_65100|data/sudoku-extreme-1k-aug-1000-test10k-v2/test|0.0|1"
)

g=0
for cell in "${GRAM_CELLS[@]}"; do
  IFS='|' read -r KEY CK ST DATA SIG NT <<< "$cell"
  ( cd "$TRM" && CUDA_VISIBLE_DEVICES=$g python experiments/residual_probe.py \
      --key "$KEY" --ckpt-dir "$CK" --step "$ST" --data "$DATA" \
      --sigma "$SIG" --n-probe 256 --n-traj "$NT" --traj-index 0 --seed 0 \
      > "$TRM/$LOG/$KEY.log" 2>&1 || echo "!!! FAIL $KEY" ) &
  g=$((g + 1))
done
wait
echo "== GRAM 8 셀 완료"

# FPRM: stochastic=False 인 S0p 도 fixed_init=False 라 시드마다 궤적이 다르다 -> 둘 다 N=50.
( cd "$FPRM" && CUDA_VISIBLE_DEVICES=0 python experiments/residual_probe.py \
    --key fprm_s0p --ckpt-dir checkpoints/fprm-nqueens10/nq10_fprm_S0p_nonoise_s0 \
    --step step_138534 --data data/nqueens-10/test \
    --n-probe 256 --n-traj 50 --traj-index 0 --traj-frames 96 --seed 0 \
    > "$TRM/$LOG/fprm_s0p.log" 2>&1 || echo "!!! FAIL fprm_s0p" ) &
( cd "$FPRM" && CUDA_VISIBLE_DEVICES=1 python experiments/residual_probe.py \
    --key fprm_s2 --ckpt-dir checkpoints/fprm-nqueens10/nq10_fprm_S2_conv3_s0 \
    --step step_138534 --data data/nqueens-10/test \
    --n-probe 256 --n-traj 50 --traj-index 0 --traj-frames 96 --seed 0 \
    > "$TRM/$LOG/fprm_s2.log" 2>&1 || echo "!!! FAIL fprm_s2" ) &
wait
echo "== 전체 10 셀 완료"
grep -h "^\[" "$TRM/$LOG"/*.log | grep -v "^\[out\]"
