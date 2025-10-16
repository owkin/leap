# Scripts

Experiment scripts for training and evaluating LEAP models.

## Pipeline Overview

LEAP uses a three-stage pipeline for drug response prediction:

1. **Pretrain Representations** (`1_pretrain_representations.py`)
   - Trains representation models (autoencoders) on gene expression data
   - Saves pretrained models to `models/` directory
   - Run once to create representation layers for downstream tasks

2. **Train Regression Heads** (`2_train_regression_heads.py`)
   - Trains prediction models on specific tasks
   - Uses pretrained representations from step 1
   - Trains multiple seeds for each task/model combination
   - Saves results to `results/` directory

3. **Ensemble Predictions** (`3_ensemble_representation_layers.py`)
   - Averages predictions across multiple representation seeds
   - Improves robustness and performance
   - Creates final ensemble models

## Quick Start

### Run Complete Pipeline

From the repository root:
```bash
bash run_pipeline.sh
```

This runs all three steps with default configurations:
- Tasks: 1
- Models: all available models
- Seeds: 0-4

### Run Individual Steps

**Step 1: Pretrain representations**
```bash
uv run --active python scripts/1_pretrain_representations.py
```

**Step 2: Train a specific model**
```bash
uv run --active python scripts/2_train_regression_heads.py \
    --task_id 1 \
    --model_id mae_ps_enet \
    --rpz_random_state 0
```

**Step 3: Create ensemble**
```bash
uv run --active python scripts/3_ensemble_representation_layers.py \
    --task_id 1 \
    --model_id mae_ps_enet
```

## Available Models

- `mae_ps_enet`: Elastic Net with perturbation-specific models
- `mae_ps_knn`: K-Nearest Neighbors with perturbation-specific models
- `mae_ps_lgbm`: LightGBM with perturbation-specific models
- `mae_pp_tdnn`: Deep neural network with pan-perturbation model
- `mae_pp_lgbm`: LightGBM with pan-perturbation model
- `mae_pp_mlp`: Multi-layer perceptron with pan-perturbation model

## Configuration

Edit variables at the top of `run_pipeline.sh`:
```bash
TASK_IDS="1"           # Space-separated task IDs
MODEL_IDS="..."        # Space-separated model IDs
SEEDS="0 1 2 3 4"      # Space-separated random seeds
```

## Important Notes

**macOS Users:** Use `uv run --active` instead of `uv run` to prevent Ray initialization issues with virtual environments.

**Results Location:** All outputs are saved to `results/` with the naming pattern:
```
results/task_{task_id}_model_{model_id}_seed_{seed}/YYYYMMDD_HHMMSS/
```

**Ensemble Location:** Ensemble results are saved to:
```
results/task_{task_id}_model_{model_id}_ensemble_seed_0/YYYYMMDD_HHMMSS/
```
