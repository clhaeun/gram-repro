# Results

Selected findings from the GRAM reproduction experiments.

| Task | Evaluation | Notes |
| --- | --- | --- |
| N-Queens 8×8 | [Evaluation JSON](experiments/results/band_nq8_v2_base_b500_s0_live_s0.json) | Multiple-sample inference; raw checkpoint |
| N-Queens 10×10 | [Evaluation JSON](experiments/results/nq10_nq10_on_b520_s0_live_s0.json) | Separate experiment configuration and evaluation |
| Graph Coloring 8 | [Evaluation JSON](experiments/results/v5gc8_gc8_v5_catnd_s0_step_58311_live_s0.json) | Multi-sample evaluation |
| Sudoku | [Diagnostic JSON](experiments/results/ema_test_sudoku_RAW.json) | EMA/raw investigation |

## Findings

- Implementation details such as KL reduction and its scaling substantially affected training behavior.
- Posterior-based diagnostics could look strong even when prior sampling performed poorly.
- EMA and raw checkpoints sometimes produced markedly different evaluation results.
- Latent noise affected the diversity and success of repeated inference.
