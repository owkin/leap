"""Define configs for regression models to use in the pipeline."""

from ml_collections import config_dict

from leap.regression_models import ElasticNet, KnnRegressor, LGBMRegressor, TorchMLPRegressor
from leap.regression_models.utils import AlphaGridElasticNet


REGRESSION_MODEL: dict[str, config_dict.ConfigDict] = {
    "knn_regressor": config_dict.ConfigDict(
        {
            "_target_": KnnRegressor,
            "n_sample_neighbors": 5,  # default
            "weights": "uniform",  # default
            "n_jobs": 30,
        }
    ),
    "elastic_net_regressor": config_dict.ConfigDict(
        {
            "_target_": ElasticNet,
            "l1_ratio": 1.0,
        }
    ),
    "lgbm_regressor": config_dict.ConfigDict(
        {
            "_target_": LGBMRegressor,
            "subsample_for_bin": 400000,
            "num_leaves": 4000,
            "min_split_gain": 0,
            "min_child_weight": 0.01,
            "min_child_samples": 5,
            "max_depth": 20,
            "learning_rate": 0.03,
            "reg_lambda": 0,
            "reg_alpha": 1,
            "colsample_bytree": 0.8,
            "n_estimators": 500,
            "subsample": 1,
            "random_state": 0,
            "n_jobs": 50,  # launch two in // on large vm
            "verbose": -1,
        }
    ),
    "lgbm_regressor_small": config_dict.ConfigDict(
        {
            # Comment every time the default is changed
            "_target_": LGBMRegressor,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 10,  # After small grid, systematically better default is -1
            "learning_rate": 0.01,  # TO TUNE, but 0.01 works well. default is 0.1
            "n_estimators": 400,  # Default is 100, but 400 is better
            "subsample_for_bin": 200000,
            "objective": None,
            "class_weight": None,
            "min_split_gain": 0,
            "min_child_weight": 1e-3,
            "min_child_samples": (5),  # Tuning it is the next best thing to do, default 20
            "subsample": 1,
            "subsample_freq": 0,
            "colsample_bytree": 0.1,  # TO TUNE, much better when small, default is 1.0
            "reg_alpha": 1,  # After small grid, better when 1, default is 0
            "reg_lambda": 1,  # After small grid, marginally better when 1, default is 0
            "random_state": 0,  # for reproducibility
            "n_jobs": 8,  # small model so we can use less cores
            "verbose": -1,  # disable prints
        }
    ),
    "mlp_regressor": config_dict.ConfigDict(
        {
            "_target_": TorchMLPRegressor,
            "hidden_layer_sizes": (512, 256, 128, 64, 32, 16),
            "activation": "relu",
            "learning_rate_init": 0.001,
            "max_epochs": 200,
            "batch_size": 2048,
            "dropout_rate": 0.2,  # Best based on tests on 1a-small
            "random_seed": 0,
            "early_stopping_use": True,
            "early_stopping_split": 0.2,
            "early_stopping_patience": 20,
            "early_stopping_delta": 0.001,
            "optimizer_type": "adam",
            "weight_decay": 1e-5,
            "learning_rate_scheduler": True,  # Best based on tests on 1a-small
            "scheduler_factor": 0.1,
            # If the threshold is the same as the delta,
            # this needs to be smaller than the patience of the early stopping
            "scheduler_patience": 10,
            "scheduler_threshold": 0.001,
            "metric": "spearman",
            "scaler_name": "robust",
            "loss_function_name": "spearman",
        }
    ),
    "mlp_regressor_small": config_dict.ConfigDict(
        {
            "_target_": TorchMLPRegressor,
            "hidden_layer_sizes": (20, 20),
            "activation": "relu",
            "learning_rate_init": 0.001,
            "max_epochs": 200,
            "batch_size": 2048,
            "dropout_rate": 0.2,  # Best based on tests on 1a-small
            "random_seed": 0,
            "early_stopping_use": True,
            "early_stopping_split": 0.2,
            "early_stopping_patience": 20,
            "early_stopping_delta": 0.001,
            "optimizer_type": "adam",
            "weight_decay": 1e-5,
            "learning_rate_scheduler": True,  # Best based on tests on 1a-small
            "scheduler_factor": 0.1,
            # If the threshold is the same as the delta,
            # this needs to be smaller than the patience of the early stopping
            "scheduler_patience": 10,
            "scheduler_threshold": 0.001,
            "metric": "spearman",
            "scaler_name": "robust",
            "loss_function_name": "spearman",
        }
    ),
    # For the ETL tDNN paper comparison
    "dnn_regressor": config_dict.ConfigDict(
        {
            "_target_": TorchMLPRegressor,
            "hidden_layer_sizes": (250, 125, 60, 30),
            "activation": "relu",
            # "The learning rate was initialized at 0.001"
            "learning_rate_init": 0.001,
            # "otherwise the full learning process would take 100 epochs"
            "max_epochs": 100,
            "batch_size": 2048,
            "dropout_rate": 0.0,
            "random_seed": 0,
            # "The learning process would be early stopped if the reduction of
            # validation loss was smaller than 0.00001 in 20 epochs"
            "early_stopping_use": True,
            "early_stopping_split": 0.2,
            "early_stopping_patience": 20,
            "early_stopping_delta": 0.00001,
            # "The Adam optimizer was used with default setting for model learning"
            "optimizer_type": "adam",
            "weight_decay": 1e-5,
            # "The learning rate [...] was reduced by a factor of 10 if the reduction of
            # validation loss was smaller than 0.00001 in 10 epochs."
            "learning_rate_scheduler": True,
            "scheduler_factor": 0.1,
            "scheduler_patience": 10,
            "scheduler_threshold": 0.00001,
            "metric": "mse",  # In the ETL paper (tDNN) it's the mse (loss)
            "scaler_name": "standard",
            "loss_function_name": "mse",
        }
    ),
}

HPT_TUNING_PARAM_GRID: dict[str, config_dict.ConfigDict | None] = {
    "knn_regressor": None,
    "elastic_net_regressor": config_dict.ConfigDict(
        {
            "alpha": config_dict.ConfigDict(
                {
                    "_target_": AlphaGridElasticNet,
                    "alpha_min_ratio": 1e-3,
                    "n_alphas": 10,
                }
            ),
        }
    ),
    "lgbm_regressor": config_dict.ConfigDict(
        {
            "reg_alpha": [0, 1],
            # log-spaced between 1e-2 and 2e-1, rounded to the first non-zero decimal
            "learning_rate": [0.01, 0.02, 0.04, 0.09, 0.2],
        }
    ),
    "lgbm_regressor_small": config_dict.ConfigDict(
        {
            "learning_rate": [0.005, 0.01],
            "colsample_bytree": [0.05, 0.1, 0.15, 0.2, 0.25],
        }
    ),
    "mlp_regressor": config_dict.ConfigDict(
        {
            # log-spaced between 5e-4 and 1e-2, rounded to the first non-zero decimal
            "learning_rate_init": [0.0005, 0.001, 0.002, 0.005, 0.01],
            "batch_size": [2048, 8192],
        }
    ),
    "mlp_regressor_small": config_dict.ConfigDict(
        {
            # log-spaced between 5e-4 and 1e-2, rounded to the first non-zero decimal
            "learning_rate_init": [0.0005, 0.001, 0.002, 0.005, 0.01],
            "hidden_layer_sizes": [
                (20,),
                (20, 20),
            ],
        }
    ),
    "dnn_regressor": config_dict.ConfigDict(
        {
            # This correspond to the HPT done in the ETL paper (tDNN)
            # "In the analysis, the dropout rate was selected among 0, 0.1, 0.25, 0.45,
            # and 0.7 by minimizing the validation loss. It was the only hyperparameter
            # optimized in the model learning process.""
            "dropout_rate": [0, 0.1, 0.25, 0.45, 0.7],
        }
    ),
}
