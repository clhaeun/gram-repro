# archive/ — frozen `gram.py` variants

`models/recursive_reasoning/gram.py` is edited in place per experiment, and checkpoints
load by class path, so the file that produced a given checkpoint is not recoverable from
git (the file is untracked). These are frozen copies.

## `gram_halt_only_as_run.py`

The exact code that trained `gram_sudoku_halt_only/step_65100` — **the best model so far**:

| metric | value |
|---|---|
| single exact, full 422,786 test | **71.85%** |
| majority@20 (n=1000, sigma=1.0) | 88.9% |
| best-of-20 by LPRM (sigma=1.0) | 87.9% |
| best-of-20 by LPRM (sigma_scale=0.5) | 93.2% |

Byte-identical to `checkpoints/Sudoku-extreme-1k-aug-1000-test10k-ACT-torch/gram_sudoku_halt_only/gram.py`
(md5 `9fe443791ee08c27a7ecc1086b0fe0a5`). Use this to reproduce the training run.

Note: an unrelated typo (`self.config.kl_balanceself`) was present in the working tree at
one point but never in this file — this copy is the good one.

## `gram_halt_only_tts.py`

Same code plus the two test-time knobs the `experiments/e1..e4` harness requires:

- `inner.sigma_scale` (float, default `1.0`) — multiplies the sampled latent std
- `use_posterior` kwarg on both `forward`s (default `None` → `self.training`) — lets an
  eval-mode rollout sample from `q(.|u,y)` without leaving eval mode, so ACT halting
  stays deterministic

Verified bit-exact against `gram_halt_only_as_run.py` at default knob values, in **both**
directions:

- eval mode: 16 ACT steps x 6 tensors (`logits`, `preds`, `lrpm_logits`, `q_halt_logits`,
  `loss`, `halted`) — 96/96 identical
- train mode: `logits`, `kl`, `loss`, and all 24 parameter gradients identical

Use this one to re-run e1–e4 against the halt_only checkpoints; `gram_halt_only_as_run.py`
lacks the knobs and the harness will fail on it.

## Restoring

Copy the wanted variant over `models/recursive_reasoning/gram.py`. Nothing imports from
`archive/` — these files are inert.
