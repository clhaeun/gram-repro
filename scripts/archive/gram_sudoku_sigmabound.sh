#!/bin/bash
# 4x RTX 3090 GRAM run: Sudoku-Extreme only
# sigma-bound variant: identical to gram_sudoku_halt_only except that gram.py bounds
# the latent std as std = 0.6*sigmoid(raw) (Dreamer/RSSM style) for BOTH prior and posterior.
set -euo pipefail
cd /workspace/TinyRecursiveModels

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs
run_name="gram_sudoku_sigmabound"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29521 --nnodes=1 \
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
