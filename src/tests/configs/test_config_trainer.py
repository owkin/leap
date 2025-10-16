"""Test config_trainer.py for consistency."""

from configs import config_trainer


class TestConfigTrainer:
    """Test that all config dictionaries are consistent."""

    def test_all_dicts_have_same_keys(self) -> None:
        """Test that all config dictionaries have the same task IDs as keys."""
        dict_names = [
            name for name in dir(config_trainer) if name.isupper() and isinstance(getattr(config_trainer, name), dict)
        ]

        if len(dict_names) < 2:
            return

        # All dicts should have the same keys
        reference_keys = set(getattr(config_trainer, dict_names[0]).keys())

        for dict_name in dict_names[1:]:
            current_dict = getattr(config_trainer, dict_name)
            assert set(current_dict.keys()) == reference_keys, (
                f"{dict_name} has keys {set(current_dict.keys())} but expected {reference_keys}"
            )

    def test_within_dict_type_consistency(self) -> None:
        """Test that within each dictionary, all values have consistent types (allowing for str/list[str] mix)."""
        dict_names = [
            name for name in dir(config_trainer) if name.isupper() and isinstance(getattr(config_trainer, name), dict)
        ]

        for dict_name in dict_names:
            current_dict = getattr(config_trainer, dict_name)
            if not current_dict:
                continue

            # Get types of all values
            value_types = {}
            for key, value in current_dict.items():
                if value is None:
                    value_types[key] = "None"
                elif isinstance(value, list):
                    value_types[key] = "list"
                elif isinstance(value, str):
                    value_types[key] = "str"
                elif isinstance(value, (int, bool)):
                    value_types[key] = type(value).__name__
                else:
                    value_types[key] = type(value).__name__

            # Check all values in this dict have compatible types
            unique_types = set(value_types.values())

            # These combinations are allowed:
            # - All same type
            # - str + list (common: single study vs multiple studies)
            # - str + None or list + None
            allowed_combos = [
                {"str"},
                {"list"},
                {"str", "list"},
                {"str", "None"},
                {"list", "None"},
                {"str", "list", "None"},
                {"int"},
                {"bool"},
            ]

            is_valid = len(unique_types) == 1 or unique_types in allowed_combos
            assert is_valid, f"{dict_name} has inconsistent value types: {value_types}"
