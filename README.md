# GRAM Reproduction

Reproduction and analysis of GRAM based on the [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) codebase.

Experiments on N-Queens, Graph Coloring, and Sudoku focused on implementation choices and evaluation behavior.

## Work

- Implemented the GRAM model and training/evaluation pipeline.
- Analyzed the effects of KL reduction, latent noise, and EMA evaluation.
- Compared single-sample and multi-sample inference, including coverage and majority voting.

## Code

- [GRAM model](models/recursive_reasoning/gram.py)
- [Training](pretrain.py) · [Configurations](config/arch/)
- [Dataset generation](dataset/) · [Evaluators](evaluators/)
- [N-Queens evaluation](experiments/sweep_harness.py) · [Graph Coloring evaluation](experiments/gc_harness.py) · [Sudoku evaluation](experiments/sudoku_harness.py)

## Results

Experimental findings and detailed measurements are recorded in [REPORT_FINAL.md](REPORT_FINAL.md), with supporting JSON files in [experiments/results/](experiments/results/).

Model checkpoints and generated datasets are not included in this repository.

## Reference

[TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
