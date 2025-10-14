"""Define perturbation model parameters."""

PRED_MODEL_NAME: dict[str, str] = {
    "mae_pp_tdnn": "dnn_regressor",
    "mae_pp_mlp": "mlp_regressor",
    "mae_pp_lgbm": "lgbm_regressor",
    "mae_ps_knn": "knn_regressor",
    "mae_ps_mlp": "mlp_regressor_small",
    "mae_ps_lgbm": "lgbm_regressor_small",
    "mae_ps_enet": "elastic_net_regressor",
}

PRED_MODEL_TYPE: dict[str, str] = {
    "mae_pp_tdnn": "pan_perturbation",
    "mae_pp_mlp": "pan_perturbation",
    "mae_pp_lgbm": "pan_perturbation",
    "mae_ps_knn": "multi_label",
    "mae_ps_mlp": "perturbation_specific",
    "mae_ps_lgbm": "perturbation_specific",
    "mae_ps_enet": "perturbation_specific",
}

RPZ_MODEL_NAME: dict[str, str] = {
    "mae_pp_tdnn": "mae",
    "mae_pp_mlp": "mae",
    "mae_pp_lgbm": "mae",
    "mae_ps_knn": "mae",
    "mae_ps_mlp": "mae",
    "mae_ps_lgbm": "mae",
    "mae_ps_enet": "mae",
}

USE_TRAINED_PREPROCESSOR: dict[str, bool] = {
    "mae_pp_tdnn": True,
    "mae_pp_mlp": True,
    "mae_pp_lgbm": True,
    "mae_ps_knn": True,
    "mae_ps_mlp": True,
    "mae_ps_lgbm": True,
    "mae_ps_enet": True,
}


USE_TRAINED_RPZ: dict[str, bool] = {
    "mae_pp_tdnn": True,
    "mae_pp_mlp": True,
    "mae_pp_lgbm": True,
    "mae_ps_knn": True,
    "mae_ps_mlp": True,
    "mae_ps_lgbm": True,
    "mae_ps_enet": True,
}

# IMPORTANT: in LEAP we actually use depmap_gdsc_pdx (using all available data)
PRETRAINED_DATA: dict[str, str] = {
    "mae_pp_tdnn": "depmap",
    "mae_pp_mlp": "depmap",
    "mae_pp_lgbm": "depmap",
    "mae_ps_knn": "depmap",
    "mae_ps_mlp": "depmap",
    "mae_ps_lgbm": "depmap",
    "mae_ps_enet": "depmap",
}


ENSEMBLING: dict[str, bool] = {
    "mae_pp_tdnn": True,
    "mae_pp_mlp": True,
    "mae_pp_lgbm": True,
    "mae_ps_knn": True,
    "mae_ps_mlp": True,
    "mae_ps_lgbm": True,
    "mae_ps_enet": True,
}

ENSEMBLING_SAVE_MODELS_TO_DISK: dict[str, bool] = {
    "mae_pp_tdnn": True,
    "mae_pp_mlp": True,
    "mae_pp_lgbm": True,
    "mae_ps_knn": False,
    "mae_ps_mlp": False,
    "mae_ps_lgbm": False,
    "mae_ps_enet": False,
}

USE_RAY: dict[str, bool] = {
    "mae_pp_tdnn": False,
    "mae_pp_mlp": False,
    "mae_pp_lgbm": False,
    "mae_ps_knn": False,
    "mae_ps_mlp": True,
    "mae_ps_lgbm": True,
    "mae_ps_enet": True,
}

RAY_REMOTE_PARAMS: dict[str, dict | None] = {
    "mae_pp_tdnn": None,
    "mae_pp_mlp": None,
    "mae_pp_lgbm": None,
    "mae_ps_knn": None,
    "mae_ps_mlp": {"num_cpus": 1, "num_gpus": 0.05},
    "mae_ps_lgbm": {"num_cpus": 8},
    "mae_ps_enet": {"num_cpus": 1},
}
