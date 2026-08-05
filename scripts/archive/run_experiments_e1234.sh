#!/bin/bash
# E1-E4 analysis sweep on GPU 4-7 (GPU 0-3 are busy with the gram_sudoku_long training run).
#
# Primary : gram_sudoku_lsum_exp / step_58590  (best of that run, exact_acc 0.7083)
# Follow-up: e1 on gram_sudoku_lsum / step_39060 (best of that run, exact_acc 0.7024)
#            -> compares failure structure between two runs at near-equal exact accuracy.
#
# Port 29503: 29501 is taken by the in-flight gram_sudoku_long training run.
set -uo pipefail
cd /workspace/TinyRecursiveModels

export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs
CKPT_ROOT="checkpoints/Sudoku-extreme-1k-aug-1000-test10k-ACT-torch"
CKPT_EXP="${CKPT_ROOT}/gram_sudoku_lsum_exp"
CKPT_LSUM="${CKPT_ROOT}/gram_sudoku_lsum"
N=1000
K=16
PORT=29503

run_exp () {
  local script="$1" step="$2" ckpt="$3" tag="$4"
  local log="logs/${tag}.log"
  echo "=================================================================="
  echo ">>> ${tag}  |  ${script}  |  ${ckpt}/${step}"
  echo "=================================================================="
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} --nnodes=1 \
    "experiments/${script}" \
    --ckpt-dir "${ckpt}" \
    --step "${step}" \
    --num-puzzles ${N} \
    --k ${K} \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  if [ ${rc} -ne 0 ]; then
    echo "!!! ${tag} FAILED (exit ${rc}) -- see ${log}; continuing to next experiment."
  else
    echo "<<< ${tag} done."
  fi
  return 0
}

# --- primary sweep: e1 -> e2 -> e3 -> e4 on the _exp best checkpoint ---
run_exp e1_failure_decomp.py step_58590 "${CKPT_EXP}"  e1_exp_58590
run_exp e2_gap.py            step_58590 "${CKPT_EXP}"  e2_exp_58590
run_exp e3_lprm.py           step_58590 "${CKPT_EXP}"  e3_exp_58590
run_exp e4_sigma.py          step_58590 "${CKPT_EXP}"  e4_exp_58590

# --- follow-up: e1 on the other run's best, for failure-structure comparison ---
run_exp e1_failure_decomp.py step_39060 "${CKPT_LSUM}" e1_lsum_39060

echo "=================================================================="
echo "ALL DONE. Logs: logs/e{1,2,3,4}_exp_58590.log logs/e1_lsum_39060.log"
echo "Outputs:  ${CKPT_EXP}/experiments/  and  ${CKPT_LSUM}/experiments/"
echo "=================================================================="
