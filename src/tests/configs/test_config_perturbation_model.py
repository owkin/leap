"""Test config_perturbation_model.py for consistency."""

from configs import config_perturbation_model


class TestConfigPerturbationModel:
    """Test that all config dictionaries are consistent."""

    def test_all_dicts_have_same_keys(self) -> None:
        """Test that all config dictionaries have the same model IDs as keys."""
        dict_names = [
            name
            for name in dir(config_perturbation_model)
            if name.isupper() and isinstance(getattr(config_perturbation_model, name), dict)
        ]

        if len(dict_names) < 2:
            return

        # All dicts should have the same keys
        reference_keys = set(getattr(config_perturbation_model, dict_names[0]).keys())

        for dict_name in dict_names[1:]:
            current_dict = getattr(config_perturbation_model, dict_name)
            assert set(current_dict.keys()) == reference_keys, (
                f"{dict_name} has keys {set(current_dict.keys())} but expected {reference_keys}"
            )

    def test_within_dict_type_consistency(self) -> None:
        """Test that within each dictionary, all values have consistent types."""
        dict_names = [
            name
            for name in dir(config_perturbation_model)
            if name.isupper() and isinstance(getattr(config_perturbation_model, name), dict)
        ]

        for dict_name in dict_names:
            current_dict = getattr(config_perturbation_model, dict_name)
            if not current_dict:
                continue

            # Get types of all values
            value_types = {}
            for key, value in current_dict.items():
                # Normalize None and dict as compatible
                if value is None or isinstance(value, dict):
                    value_types[key] = "dict_or_none"
                else:
                    value_types[key] = type(value).__name__

            # Check all values in this dict have the same type
            unique_types = set(value_types.values())
            assert len(unique_types) == 1, f"{dict_name} has inconsistent value types: {value_types}"
