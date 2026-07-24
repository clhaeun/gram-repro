# Rollback guide

Baseline commit: `c011037` ("Clarify update message in README") — the pristine upstream state.
Saved diff of all code edits: `.rollback/local_code_changes.patch`

## What was changed

**Modified (tracked by git):**
- `README.md` — torch install index `cu126` → `cu124` (matches this box's nvcc 12.4)
- `pretrain.py` — added stdout logging only; training logic untouched:
  - train loop: `[step N] ...` line every 50 steps
  - eval branch: `[EVAL @ step N] ...` line (eval metrics previously went to wandb only, so
    a `WANDB_MODE=disabled` run silently discarded them)
  - `flatten_metrics()` helper (evaluate() returns a nested `{set: {metric: value}}` dict)

**Created (untracked):**
- `eval_checkpoint.py` — eval-only rerun on a saved checkpoint
- `analyze_sudoku_tta.py` — no-hit / pass@k vs majority-vote failure analysis
- `run_sudoku_mlp_t_4gpu.sh`, `run_eval_final.sh` — launch scripts

**Expensive artifacts — do NOT delete when rolling back:**
- `checkpoints/` — 10 checkpoints from the 7h54m 4-GPU run (192MB)
- `data/sudoku-extreme-1k-aug-1000/` — built dataset (1.3GB)
- `logs/` — training + eval logs

## Roll back code only (keeps checkpoints, data, logs)

    cd /workspace/TinyRecursiveModels
    git checkout -- README.md pretrain.py
    rm -f eval_checkpoint.py analyze_sudoku_tta.py run_sudoku_mlp_t_4gpu.sh run_eval_final.sh

## Reapply the edits later

    git apply .rollback/local_code_changes.patch

## Inspect before deciding

    git diff                    # current edits vs baseline
    git status --porcelain      # what is modified / untracked
