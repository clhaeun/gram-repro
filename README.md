# GRAM Reproduction

Reproduction and analysis of GRAM, a recursive reasoning model, based on the [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) codebase.

Experiments cover N-Queens, Graph Coloring, and Sudoku, with a focus on implementation details and evaluation behavior.

## Scope

- Reproduced GRAM training and evaluation pipelines.
- Investigated KL reduction, latent noise, and EMA checkpoint behavior.
- Evaluated single-sample accuracy and multi-sample inference.

## Repository

- `models/recursive_reasoning/gram.py` — GRAM implementation
- `config/arch/` — model configurations
- `dataset/`, `evaluators/` — datasets and task evaluation
- `experiments/` — evaluation scripts and results
- `REPORT_FINAL.md` — experiment summary and findings

## References

- [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [GRAM experiment summary](REPORT_FINAL.md)
