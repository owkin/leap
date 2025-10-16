"""Config for the one model per perturbation to be used in the prediction pipeline."""

from pathlib import Path

from loguru import logger
from ml_collections import config_dict

from configs import config_regression_model, config_rpz_model
from leap.data.preprocessor import OmicsPreprocessor
from leap.data.splits import cv_split_generator
from leap.pipelines.perturbation_pipeline import PerturbationPipeline
from leap.representation_models import PCA


# Define paths for pretrained models (relative to repository root)
REPO_PATH = Path(__file__).parent.parent
PREPROCESSOR_PATH = REPO_PATH / "models" / "preprocessors"
RPZ_PATH = REPO_PATH / "models" / "rpz"


def get_config_model(
    pred_model_type: str,
    pred_model_name: str,
    list_of_genes: str,
    normalization: str,
    rpz_model_name: str,
    use_trained_preprocessor: bool = False,
    use_trained_rpz: bool = False,
    pretrained_data: str = "depmap",
    rpz_random_state: int = 42,
    fgps_dim: int = 500,
    ensembling: bool = True,
    ensembling_save_models_to_disk: bool = False,
    use_ray: bool = False,
    ray_remote_params: dict | None = None,
) -> config_dict.ConfigDict:
    """Set the model configuration.

    Parameters
    ----------
    pred_model_type : str
        The type of model to use. Possible values are: "multi_label", "perturbation_specific", or "pan_perturbation".
    pred_model_name : str
        Name of the prediction model.
    list_of_genes : str
        The list of genes used in the trained preprocessor and rpz for rnaseq.
    normalization : str, optional
        The normalization method to use for rnaseq data, by default "tpm".
    rpz_model_name : str, optional
        The name of the trained rpz to use for rnaseq data, by default pca.
    use_trained_preprocessor : bool, optional
        Whether to use a trained preprocessor for RNASeq that is already saved, by default False.
    use_trained_rpz : bool, optional
        Whether to use a trained rpz for RNASeq that is already saved, by default False.
    pretrained_data : str, optional
        The name of the pretrained data to use, by default "depmap".
    rpz_random_state : int, optional
        The random state to use for the rpz model, by default 42.
    fgps_dim : int, optional
        The dimension of the fingerprint rpz model, by default 500.
    ensembling : bool, optional
        Whether to use ensembling, by default True.
    ensembling_save_models_to_disk : bool, optional
        Whether to save the ensembling models to disk, this can save RAM and prevent out of memory errors for heavy
        models, by default False.
    use_ray : bool, optional
        Whether to use Ray to parallelise over the perturbations. Only possible if one_model_per_perturbation is True.
        Default is False.
    ray_remote_params : dict | None, optional
        Parameters for Ray remote. Defaults to None.

    Raises
    ------
    ValueError
        If pred_model_type is not one of ("multi_label", "perturbation_specific",
        "pan_perturbation").

    Returns
    -------
    config : config_dict.ConfigDict
        The model configuration.
    """
    # temp, using the rnaseq form 23Q4
    if use_ray and pred_model_type != "perturbation_specific":
        logger.error("Ray can only be used for one model per perturbation.")
    if pred_model_type == "multi_label":
        return get_config_one_model_all_perturbations_multi_label(
            pred_model_name=pred_model_name,
            list_of_genes=list_of_genes,
            normalization=normalization,
            rpz_model_name=rpz_model_name,
            use_trained_preprocessor=use_trained_preprocessor,
            use_trained_rpz=use_trained_rpz,
            pretrained_data=pretrained_data,
            rpz_random_state=rpz_random_state,
            ensembling=ensembling,
            ensembling_save_models_to_disk=ensembling_save_models_to_disk,
        )
    if pred_model_type == "perturbation_specific":
        return get_config_one_model_per_perturbation(
            pred_model_name=pred_model_name,
            list_of_genes=list_of_genes,
            normalization=normalization,
            rpz_model_name=rpz_model_name,
            use_trained_preprocessor=use_trained_preprocessor,
            use_trained_rpz=use_trained_rpz,
            pretrained_data=pretrained_data,
            rpz_random_state=rpz_random_state,
            fgps_dim=fgps_dim,
            ensembling=ensembling,
            ensembling_save_models_to_disk=ensembling_save_models_to_disk,
            use_ray=use_ray,
            ray_remote_params=ray_remote_params,
        )
    if pred_model_type == "pan_perturbation":
        return get_config_one_model_all_perturbations_single_label(
            pred_model_name=pred_model_name,
            list_of_genes=list_of_genes,
            normalization=normalization,
            rpz_model_name=rpz_model_name,
            use_trained_preprocessor=use_trained_preprocessor,
            use_trained_rpz=use_trained_rpz,
            pretrained_data=pretrained_data,
            rpz_random_state=rpz_random_state,
            fgps_dim=fgps_dim,
            ensembling=ensembling,
            ensembling_save_models_to_disk=ensembling_save_models_to_disk,
        )
    raise ValueError(f"Invalid pred_model_type: {pred_model_type}")


def _config_backbone(
    pred_model_name: str,
    list_of_genes: str,
    normalization: str,
    rpz_model_name: str,
    use_trained_preprocessor: bool,
    use_trained_rpz: bool,
    pretrained_data: str,
    rpz_random_state: int,
    fgps_dim: int,
    ensembling: bool,
    ensembling_save_models_to_disk: bool,
) -> config_dict.ConfigDict:
    """Generate the part of the config which is common to all model configs.

    Please refer to the get_config_model docstrings for parameter descriptions.
    """
    # Initialise a default perturbation model config
    config = config_dict.ConfigDict(
        {
            "_target_": PerturbationPipeline,
            "ensembling": ensembling,
            "ensembling_save_models_to_disk": ensembling_save_models_to_disk,
            "fgpt_rpz_model": config_dict.ConfigDict(
                {
                    "_target_": PCA,
                    "repr_dim": fgps_dim,
                }
            ),
            "use_ray": False,
        }
    )

    config.hpt_tuning_cv_split = config_dict.ConfigDict(
        {
            "_target_": cv_split_generator,
            "_partial_": True,
            "k_fold": True,
            "group_variable": None,
            "leave_one_group_out": False,
            "test_split_ratio": None,
            "n_splits": 5,
            "random_state": 0,
        }
    )

    # Define preprocessor
    if use_trained_preprocessor:
        config.preprocessor_model_rnaseq = PREPROCESSOR_PATH / (
            f"log_mean_std_{pretrained_data}_{list_of_genes}_{normalization}_seed_{rpz_random_state}.pkl"
        )
    else:
        config.preprocessor_model_rnaseq = config_dict.ConfigDict(
            {
                "_target_": OmicsPreprocessor,
                "scaling_method": "mean_std",
                "max_genes": -1,
                "gene_list_source": None,
                "log_scaling": True,  # log-scaling is typically done during normalization
            }
        )

    # Define rpz model
    if rpz_model_name is None or rpz_model_name == "identity":
        config.rpz_model_rnaseq = None
    elif use_trained_rpz:
        config.rpz_model_rnaseq = RPZ_PATH / (
            f"{rpz_model_name}_{pretrained_data}_{list_of_genes}_{normalization}_seed_{rpz_random_state}.pkl"
        )
    else:
        config.rpz_model_rnaseq = config_rpz_model.RPZ_MODEL[rpz_model_name]
        config.rpz_model_rnaseq.random_state = rpz_random_state

    # Define the regression model configuration
    config.regression_model_base_instance = config_regression_model.REGRESSION_MODEL[pred_model_name]
    config.hpt_tuning_param_grid = config_regression_model.HPT_TUNING_PARAM_GRID[pred_model_name]

    return config


def get_config_one_model_per_perturbation(
    pred_model_name: str,
    list_of_genes: str,
    normalization: str,
    rpz_model_name: str,
    use_trained_preprocessor: bool,
    use_trained_rpz: bool,
    pretrained_data: str,
    rpz_random_state: int,
    fgps_dim: int,
    ensembling: bool,
    ensembling_save_models_to_disk: bool,
    use_ray: bool,
    ray_remote_params: dict | None,
) -> config_dict.ConfigDict:
    """Model configuration for one model per perturbation (perturbation-specific models).

    Please refer to the get_config_model docstrings for parameter descriptions.
    """
    config = _config_backbone(
        pred_model_name=pred_model_name,
        list_of_genes=list_of_genes,
        normalization=normalization,
        rpz_model_name=rpz_model_name,
        use_trained_preprocessor=use_trained_preprocessor,
        use_trained_rpz=use_trained_rpz,
        pretrained_data=pretrained_data,
        rpz_random_state=rpz_random_state,
        fgps_dim=fgps_dim,
        ensembling=ensembling,
        ensembling_save_models_to_disk=ensembling_save_models_to_disk,
    )

    # Define the tuning metric
    config.hpt_tuning_score = "auc" if "classifier" in pred_model_name else "spearman"
    config.one_model_per_perturbation = True

    # Use Ray to parallelise over perturbations
    config.use_ray = use_ray
    config.ray_remote_params = ray_remote_params

    return config


def get_config_one_model_all_perturbations_single_label(
    pred_model_name: str,
    list_of_genes: str,
    normalization: str,
    rpz_model_name: str,
    use_trained_preprocessor: bool,
    use_trained_rpz: bool,
    pretrained_data: str,
    rpz_random_state: int,
    fgps_dim: int,
    ensembling: bool,
    ensembling_save_models_to_disk: bool,
) -> config_dict.ConfigDict:
    """Model configuration for a pan-perturbation model (single label per sample).

    Please refer to the get_config_model docstrings for parameter descriptions.
    """
    config = _config_backbone(
        pred_model_name=pred_model_name,
        list_of_genes=list_of_genes,
        normalization=normalization,
        rpz_model_name=rpz_model_name,
        use_trained_preprocessor=use_trained_preprocessor,
        use_trained_rpz=use_trained_rpz,
        pretrained_data=pretrained_data,
        rpz_random_state=rpz_random_state,
        fgps_dim=fgps_dim,
        ensembling=ensembling,
        ensembling_save_models_to_disk=ensembling_save_models_to_disk,
    )

    # Define the tuning metric
    config.hpt_tuning_score = "auc" if "classifier" in pred_model_name else "spearman"
    config.one_model_per_perturbation = False

    # Group by sample for pan-perturbation models
    config.hpt_tuning_cv_split.group_variable = "sample"
    return config


def get_config_one_model_all_perturbations_multi_label(
    pred_model_name: str,
    list_of_genes: str,
    normalization: str,
    rpz_model_name: str,
    use_trained_preprocessor: bool,
    use_trained_rpz: bool,
    pretrained_data: str,
    rpz_random_state: int,
    ensembling: bool,
    ensembling_save_models_to_disk: bool,
) -> config_dict.ConfigDict:
    """Model configuration for a pan-perturbation model with multi-labels.

    This framework is adapted for multilabel prediction (e.g., KNN regressor).
    Memory and time efficient compared to the perturbation-specific framework.

    Please refer to the get_config_model docstrings for parameter descriptions.
    """
    config = _config_backbone(
        pred_model_name=pred_model_name,
        list_of_genes=list_of_genes,
        normalization=normalization,
        rpz_model_name=rpz_model_name,
        use_trained_preprocessor=use_trained_preprocessor,
        use_trained_rpz=use_trained_rpz,
        pretrained_data=pretrained_data,
        rpz_random_state=rpz_random_state,
        fgps_dim=0,  # Not used for multi-label models
        ensembling=ensembling,
        ensembling_save_models_to_disk=ensembling_save_models_to_disk,
    )
    config.one_model_per_perturbation = False
    config.hpt_tuning_score = None
    config.hpt_tuning_cv_split = None
    return config
