"""Full trainer config."""

import copy
from pathlib import Path

from ml_collections import config_dict

from configs import config_perturbation_model, config_trainer
from configs.get_config_data import get_config_data
from configs.get_config_models import get_config_model
from configs.get_config_split import get_config_split


REPO_PATH = Path(__file__).parent.parent


def get_config(
    task_id: str,
    model_id: str,
    rpz_random_state: int,
) -> config_dict.ConfigDict:
    """Create the full trainer configuration.

    Parameters
    ----------
    task_id : str
        The task id. Possible values are in config_trainer dictionaries.
    model_id : str
        The model id. Possible values are the keys in the config_perturbation_model dictionaries. For example,
        "mae_ps_enet" is the perturbation model_id that uses mae rpz, one model per perturbation, and the enet as a
        regression model.
    rpz_random_state : int
        The random state to use for the RPZ model.

    Raises
    ------
    ValueError
        If the task_id is not recognized.
        If the model_id is not recognized.
        If the filter_available_fingerprints is False but a pan-perturbation model is used.

    Returns
    -------
    config : config_dict.ConfigDict
        The full trainer configuration.
    """
    # Extract target tissue name for tasks on target tissues
    if task_id[0] in {"3", "4"}:
        target_tissue = "_".join(task_id.split("_")[1:])
        task_id = task_id.split("_")[0]

    # Check that the task_id is recognized
    if task_id not in config_trainer.SOURCE_DOMAIN_STUDIES:
        raise ValueError(f"Task id {task_id} is not recognized.")

    # Check that the model_id is recognized
    if model_id not in config_perturbation_model.PRED_MODEL_TYPE:
        raise ValueError(f"Model id {model_id} is not recognized.")

    # Get the data configuration
    config = get_config_data(
        source_domain_studies=config_trainer.SOURCE_DOMAIN_STUDIES[task_id],
        source_domain_label=config_trainer.SOURCE_DOMAIN_LABEL[task_id],
        list_of_perturbations=config_trainer.LIST_OF_PERTURBATIONS[task_id],
        filter_available_fingerprints=config_trainer.FILTER_AVAILABLE_FINGERPRINTS[task_id],
        normalization=config_trainer.NORMALIZATION[task_id],
        list_of_genes=config_trainer.LIST_OF_GENES[task_id],
        target_domain_studies=config_trainer.TARGET_DOMAIN_STUDIES[task_id],
        target_domain_label=config_trainer.TARGET_DOMAIN_LABEL[task_id],
    )

    # Split the data into source and target domains for tasks on target tissues
    if task_id[0] in {"3", "4"}:
        if target_tissue != "":
            config.target_domain_data = copy.deepcopy(config.source_domain_data)
            # The source data includes all tissues except the target tissue
            config.source_domain_data.tissues_to_exclude = [target_tissue]
            # The target data includes only the target tissue
            config.target_domain_data.tissues_to_keep = [target_tissue]
            config.target_domain_data.min_n_label = 0  # enough for test set

    # Get the training/few-shot/test split configuration
    config.data_split = get_config_split(test_set_type=config_trainer.TEST_SET_TYPE[task_id])

    # Define the type of perturbation model
    if (
        config_perturbation_model.PRED_MODEL_TYPE[model_id] == "pan_perturbation"
        and not config_trainer.FILTER_AVAILABLE_FINGERPRINTS[task_id]
    ):
        raise ValueError("Filter available fingerprints must be True for a pan-perturbation model.")

    # Get the perturbation model configuration
    config.model = get_config_model(
        pred_model_type=config_perturbation_model.PRED_MODEL_TYPE[model_id],
        pred_model_name=config_perturbation_model.PRED_MODEL_NAME[model_id],
        list_of_genes=config_trainer.LIST_OF_GENES[task_id],
        normalization=config_trainer.NORMALIZATION[task_id],
        rpz_model_name=config_perturbation_model.RPZ_MODEL_NAME[model_id],
        use_trained_preprocessor=config_perturbation_model.USE_TRAINED_PREPROCESSOR[model_id],
        use_trained_rpz=config_perturbation_model.USE_TRAINED_RPZ[model_id],
        pretrained_data=config_perturbation_model.PRETRAINED_DATA[model_id],
        rpz_random_state=rpz_random_state,  # script argument
        fgps_dim=config_trainer.FGPS_DIM[task_id],
        ensembling=config_perturbation_model.ENSEMBLING[model_id],
        ensembling_save_models_to_disk=(config_perturbation_model.ENSEMBLING_SAVE_MODELS_TO_DISK[model_id]),
        use_ray=config_perturbation_model.USE_RAY[model_id],
        ray_remote_params=config_perturbation_model.RAY_REMOTE_PARAMS[model_id],
    )

    return config
