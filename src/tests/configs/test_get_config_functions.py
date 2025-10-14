"""Test get_config_*.py functions return valid configurations."""

import inspect

import pytest

from configs.get_config import get_config
from configs.get_config_data import get_config_data
from configs.get_config_models import get_config_model
from configs.get_config_split import get_config_split


def check_config_params_valid(config, config_name="config"):
    """Recursively check that config parameters match the _target_ class signature.

    This catches issues where config has parameters the target class doesn't accept.
    """
    if not hasattr(config, "_target_"):
        return

    target = config["_target_"]

    # Get signature
    try:
        if inspect.isclass(target):
            sig = inspect.signature(target.__init__)
        else:
            sig = inspect.signature(target)
    except (ValueError, TypeError):
        return

    # Get valid parameter names (excluding self)
    valid_params = {p for p in sig.parameters.keys() if p != "self"}

    # Check if target accepts **kwargs
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    # Get config parameters (excluding _target_ and _partial_)
    config_params = {k for k in config.keys() if not k.startswith("_")}

    # If no **kwargs, check for invalid parameters
    if not has_var_keyword:
        invalid_params = config_params - valid_params
        assert not invalid_params, (
            f"{config_name} (target={target.__name__}) has invalid parameters: {invalid_params}. "
            f"Valid parameters are: {sorted(valid_params)}"
        )


class TestGetConfigData:
    """Test get_config_data functions."""

    def test_basic_config_generation(self) -> None:
        """Test that get_config_data generates valid configs."""
        config = get_config_data(
            source_domain_studies="DepMap_23Q4",
            source_domain_label="gene_dependency",
            list_of_perturbations="perturbations_task_1",
            filter_available_fingerprints=True,
            normalization="tpm",
            list_of_genes="most_variant_genes",
        )

        assert "source_domain_data" in config
        assert "_target_" in config.source_domain_data
        check_config_params_valid(config.source_domain_data, "source_domain_data")


class TestGetConfigSplit:
    """Test get_config_split functions."""

    @pytest.mark.parametrize("test_set_type", ["sample", "perturbation", "tissue", "transfer_learning"])
    def test_split_config_generation(self, test_set_type: str) -> None:
        """Test that get_config_split generates valid configs."""
        config = get_config_split(test_set_type=test_set_type, training_split_count=10)

        assert "_target_" in config
        assert "_partial_" in config
        # Partial configs will be called with additional args later, so we can't validate them fully


class TestGetConfigModels:
    """Test get_config_models functions."""

    @pytest.mark.parametrize("pred_model_type", ["multi_label", "perturbation_specific", "pan_perturbation"])
    def test_model_config_generation(self, pred_model_type: str) -> None:
        """Test that get_config_model generates valid configs."""
        pred_model_name = "knn_regressor" if pred_model_type == "multi_label" else "elastic_net_regressor"

        config = get_config_model(
            pred_model_type=pred_model_type,
            pred_model_name=pred_model_name,
            list_of_genes="most_variant_genes",
            normalization="tpm",
            rpz_model_name="pca",
            use_trained_preprocessor=False,
            use_trained_rpz=False,
            pretrained_data="depmap",
            rpz_random_state=0,
            fgps_dim=256,
            ensembling=True,
            ensembling_save_models_to_disk=False,
            use_ray=False,
            ray_remote_params=None,
        )

        assert "_target_" in config
        check_config_params_valid(config, "model")


class TestGetConfig:
    """Test the full get_config function."""

    def test_full_config_generation(self) -> None:
        """Test that get_config generates a complete valid config."""
        config = get_config(task_id="1", model_id="mae_ps_enet", rpz_random_state=0)

        # Check main structure
        assert "source_domain_data" in config
        assert "data_split" in config
        assert "model" in config

        # Check all have _target_
        assert "_target_" in config.source_domain_data
        assert "_target_" in config.data_split
        assert "_target_" in config.model

        # Validate parameter signatures
        check_config_params_valid(config.source_domain_data, "source_domain_data")
        check_config_params_valid(config.model, "model")

    def test_invalid_task_id(self) -> None:
        """Test that invalid task_id raises error."""
        with pytest.raises(ValueError, match="not recognized"):
            get_config(task_id="invalid", model_id="mae_ps_enet", rpz_random_state=0)

    def test_invalid_model_id(self) -> None:
        """Test that invalid model_id raises error."""
        with pytest.raises(ValueError, match="not recognized"):
            get_config(task_id="1", model_id="invalid", rpz_random_state=0)
