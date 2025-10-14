"""Test config_regression_model.py for valid ConfigDict structures."""

from ml_collections import config_dict

from configs import config_regression_model


class TestConfigRegressionModel:
    """Test regression model configurations."""

    def test_all_main_dicts_have_same_keys(self) -> None:
        """Test that REGRESSION_MODEL and HPT_TUNING_PARAM_GRID have same model keys."""
        regression_keys = set(config_regression_model.REGRESSION_MODEL.keys())
        hpt_keys = set(config_regression_model.HPT_TUNING_PARAM_GRID.keys())

        assert regression_keys == hpt_keys, (
            f"Model keys should match across all dicts. "
            f"REGRESSION_MODEL: {regression_keys}, "
            f"HPT_TUNING_PARAM_GRID: {hpt_keys}, "
        )

    def test_all_regression_configs_have_target(self) -> None:
        """Test that all regression model configs have _target_ key."""
        for key, value in config_regression_model.REGRESSION_MODEL.items():
            assert isinstance(value, config_dict.ConfigDict), f"{key} should be a ConfigDict"
            assert "_target_" in value, f"{key} should have '_target_' key"
            assert callable(value["_target_"]), f"{key}['_target_'] should be a class"

    def test_mlp_variants_have_same_structure(self) -> None:
        """Test that MLP model variants have the same parameter structure."""
        mlp_models = [k for k in config_regression_model.REGRESSION_MODEL.keys() if "mlp" in k or "dnn" in k]

        if len(mlp_models) < 2:
            return

        # Get parameter sets for each MLP model
        param_sets = {}
        for model in mlp_models:
            param_sets[model] = set(config_regression_model.REGRESSION_MODEL[model].keys()) - {"_target_"}

        # Check they all have the same parameters
        reference_params = param_sets[mlp_models[0]]
        for model in mlp_models[1:]:
            assert param_sets[model] == reference_params, (
                f"{model} and {mlp_models[0]} should have same parameters. "
                f"Difference: {param_sets[model].symmetric_difference(reference_params)}"
            )
