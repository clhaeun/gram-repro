#!/bin/bash
# Controlled clip experiment: identical to gram_sudoku_lsum_exp.sh except for
# grad-norm clipping (now unconditional in pretrain.py) and a denser eval grid.
#
#   gram_sudoku_lsum_exp : 50000 epochs, no clip   -> peak 0.7083 @ step_58590, then collapse
#   gram_sudoku_clip50k  : 50000 epochs, clip 1.0  -> isolates the clip effect
#
# Run A (gram_sudoku_long) confounds clip with a 2x longer schedule; this run does not.
# NOTE: eval_interval 5000 -> 2000 differs from the baseline, but only changes measurement
#       cadence, not optimization. See the report for the caveat about checkpoint volume.
set -euo pipefail
cd /workspace/TinyRecursiveModels

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs
run_name="gram_sudoku_clip50k"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29501 --nnodes=1 \
  pretrain.py \
  arch=gram \
  data_paths="[data/sudoku-extreme-1k-aug-1000-test10k]" \
  evaluators="[]" \
  epochs=50000 eval_interval=2000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} ema=True ema_rate=0.9999 \
  2>&1 | tee -a logs/${run_name}.log
