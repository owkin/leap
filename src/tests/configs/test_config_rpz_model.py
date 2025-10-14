"""Test config_rpz_model.py for valid ConfigDict structures."""

from ml_collections import config_dict

from configs import config_rpz_model


class TestConfigRpzModel:
    """Test RPZ_MODEL configuration."""

    def test_all_configs_have_target(self) -> None:
        """Test that all configs have _target_ key."""
        for key, value in config_rpz_model.RPZ_MODEL.items():
            assert isinstance(value, config_dict.ConfigDict), f"{key} should be a ConfigDict"
            assert "_target_" in value, f"{key} should have '_target_' key"
            assert callable(value["_target_"]), f"{key}['_target_'] should be a class or function"

    def test_mae_and_ae_have_same_parameters(self) -> None:
        """Test that MAE and AE configs have the same parameter structure."""
        if "mae" not in config_rpz_model.RPZ_MODEL or "ae" not in config_rpz_model.RPZ_MODEL:
            return

        mae_keys = set(config_rpz_model.RPZ_MODEL["mae"].keys()) - {"_target_"}
        ae_keys = set(config_rpz_model.RPZ_MODEL["ae"].keys()) - {"_target_"}

        assert mae_keys == ae_keys, (
            f"MAE and AE should have same parameters. MAE only: {mae_keys - ae_keys}, AE only: {ae_keys - mae_keys}"
        )
