#!/usr/bin/env bash
set -euo pipefail
cd /workspace/TinyRecursiveModels
export TORCH_CUDA_ARCH_LIST="8.6" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p logs
CKPT_DIR=checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  eval_checkpoint.py "$CKPT_DIR" step_65100 2>&1 | tee logs/eval_step_65100.log
