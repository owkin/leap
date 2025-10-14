# LEAP: Layered Ensemble of Autoencoders and Predictors

**Predicting gene essentiality and drug response from perturbation screens in preclinical cancer models**

[![arXiv](https://img.shields.io/badge/arXiv-2502.15646-b31b1b.svg)](https://arxiv.org/abs/2502.15646)
[![Coverage](./badges/cov_badge.svg)](./badges/cov_badge.svg)
[![CI/CD](https://github.com/owkin/leap/actions/workflows/ci-cd.yaml/badge.svg)](https://github.com/owkin/leap/actions/workflows/ci-cd.yaml)

## Overview

LEAP is a novel ensemble framework designed to improve robustness and generalization in predicting perturbation responses from molecular profiles. By leveraging multiple DAMAE (Data Augmented Masked Autoencoder) representations and LASSO regressors, LEAP consistently outperforms state-of-the-art approaches in predicting gene essentiality and drug responses across diverse biological contexts.

## Installation

```bash
make install
```

This will install `uv` (if needed) and all project dependencies.

## Contributing

Run quality checks before committing:

```bash
make checks   # Run pre-commit hooks (linting, formatting, type checking)
make tests  # Run tests with coverage
```

## Usage

### Clean up temporary files
```bash
make clean
```
Removes all temporary files, caches, and build artifacts.

### Clean up data and results
```bash
make clean-data
```
Removes all processed data, trained models, and results. **Warning: This action cannot be undone!**

### Run the complete LEAP pipeline
```bash
sh run_pipeline.sh
```
Runs the full LEAP pipeline end-to-end:
1. Pretrain representations
2. Train regression heads for multiple tasks, models, and seeds
3. Ensemble predictions

The pipeline is configured to run on task 1 with the `mae_ps_enet` model across 5 different seeds.

## Citation

If you use LEAP in your research, please cite our paper:

```bibtex
@article{bodinier2025predicting,
  title={Predicting gene essentiality and drug response from perturbation screens in preclinical cancer models with LEAP: Layered Ensemble of Autoencoders and Predictors},
  author={Bodinier, Barbara and Dissez, Gaetan and Bleistein, Linus and Dauvin, Antonin},
  journal={arXiv preprint arXiv:2502.15646},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Barbara Bodinier
- Gaetan Dissez
- Lucile Ter-Minassian
- Linus Bleistein
- Roberta Codato
- John Klein
- Eric Durand
- Antonin Dauvin
