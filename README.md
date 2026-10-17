# GRAM Reproduction

Reproduction and analysis of GRAM based on [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

Experiments cover N-Queens, Graph Coloring, and Sudoku, focusing on implementation choices, stochastic inference, and evaluation.

## Work

- Reproduced the GRAM model and training/evaluation pipeline.
- Investigated KL reduction, latent noise, and EMA/raw checkpoint behavior.
- Compared single-sample and multi-sample evaluation.

## Code

- [GRAM model](models/recursive_reasoning/gram.py) · [Training](pretrain.py)
- [Configurations](config/arch/) · [Datasets](dataset/) · [Evaluators](evaluators/)
- [N-Queens](experiments/sweep_harness.py) · [Graph Coloring](experiments/gc_harness.py) · [Sudoku](experiments/sudoku_harness.py)

## Results

See [RESULTS.md](RESULTS.md) for selected findings and [experiments/results/](experiments/results/) for example evaluation outputs.

Checkpoints and generated datasets are not included. The result JSON files document previous evaluations and are not sufficient on their own to rerun training.

## Reference

[TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
