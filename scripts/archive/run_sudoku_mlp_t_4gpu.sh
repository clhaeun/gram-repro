#!/usr/bin/env bash
# 4x RTX 3090 full run: pretrain_mlp_t_sudoku (target ~87% exact-accuracy)
# global_batch_size stays at default 768 -> per-GPU 192 (validated ~8GB/24GB).
set -euo pipefail
cd /workspace/TinyRecursiveModels

export TORCH_CUDA_ARCH_LIST="8.6"                       # RTX 3090 = sm_86
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled                              # no wandb for this checkpoint run
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs
run_name="pretrain_mlp_t_sudoku"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} ema=True \
  2>&1 | tee -a logs/${run_name}.log
