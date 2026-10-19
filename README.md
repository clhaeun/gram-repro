# GRAM Reproduction

Reproduction and experimental analysis of GRAM, built from the [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) codebase. This project was conducted during a research internship at KAIST MLML.

The work focuses on **reproduction and diagnosis**, rather than proposing a new reasoning model.

## Questions investigated

- How do implementation choices left underspecified in the paper, including KL reduction and its scaling, affect training?
- How do latent noise and multiple inference samples influence valid solutions and solution coverage?
- How do prior-based evaluation and posterior diagnostics differ?
- When do raw and EMA checkpoints give different evaluation results?

Experiments cover N-Queens, Graph Coloring, and Sudoku. Reported observations apply to the configurations and evaluations used in this reproduction, not necessarily to GRAM in general.

## Implementation

| Component | Source |
| --- | --- |
| Model | [`models/recursive_reasoning/gram.py`](models/recursive_reasoning/gram.py) |
| Training | [`pretrain.py`](pretrain.py) |
| Configurations | [`config/arch/`](config/arch/) |
| Dataset generation and evaluation | [`dataset/`](dataset/), [`evaluators/`](evaluators/) |
| Experiment harnesses | [N-Queens](experiments/sweep_harness.py), [Graph Coloring](experiments/gc_harness.py), [Sudoku](experiments/sudoku_harness.py) |

## Results

[Selected observations and evaluation outputs](RESULTS.md) · [Representative JSON files](experiments/results/)

The reproduction exposed sensitivity to several implementation choices and differences between sample generation and candidate selection. Some comparisons with published numbers use different checkpoint/seed selection or unresolved metric conventions; see the notes accompanying each result before treating values as directly comparable.

Checkpoints and generated training datasets are not included, so the selected JSON outputs **do not** constitute a complete end-to-end reproduction package.

## Reference

[SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
