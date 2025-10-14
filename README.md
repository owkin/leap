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

<!-- Usage instructions and examples coming soon -->

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
