#!/bin/bash
# LEAP Pipeline: Pretrain representations, train models, and ensemble results
set -e  # Exit on error

# Configuration
TASK_IDS="1"
# MODEL_IDS="mae_ps_knn mae_ps_enet mae_pp_tdnn mae_ps_lgbm mae_pp_lgbm mae_pp_mlp"
MODEL_IDS="mae_ps_enet"
SEEDS="0 1 2 3 4"

echo "=== LEAP Pipeline ==="
echo "Tasks: $TASK_IDS"
echo "Models: $MODEL_IDS"
echo "Seeds: $SEEDS"
echo ""

# Step 1: Pretrain representations
echo "[1/3] Pretraining representations..."
uv run python scripts/1_pretrain_representations.py

# Step 2: Train regression heads
echo "[2/3] Training regression heads..."
for task in $TASK_IDS; do
    for model in $MODEL_IDS; do
        for seed in $SEEDS; do
            echo "  Training: task=$task, model=$model, seed=$seed"
            uv run --active python scripts/2_train_regression_heads.py \
                --task_id $task \
                --model_id $model \
                --rpz_random_state $seed
        done
    done
done

# Step 3: Ensemble predictions
echo "[3/3] Ensembling predictions..."
for task in $TASK_IDS; do
    for model in $MODEL_IDS; do
        echo "  Ensembling: task=$task, model=$model"
        uv run python scripts/3_ensemble_representation_layers.py \
            --task_id $task \
            --model_id $model
    done
done

echo "✓ Pipeline complete!"
