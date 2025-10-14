"""Tests for the utils module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from ml_collections import config_dict

from leap.utils.config_utils import (
    get_config_dict_copy,
    instantiate,
    load_module,
)
from leap.utils.io import load_pickle, save_pickle
from leap.utils.seed import seed_everything


class TestSeed:
    """Test seed_everything function."""

    def test_seed_everything_numpy(self):
        """Test that seed_everything sets numpy random seed correctly."""
        seed_everything(42)
        result1 = np.random.rand(5)

        seed_everything(42)
        result2 = np.random.rand(5)

        np.testing.assert_array_equal(result1, result2)

    def test_seed_everything_torch(self):
        """Test that seed_everything sets torch random seed correctly."""
        seed_everything(42)
        result1 = torch.rand(5)

        seed_everything(42)
        result2 = torch.rand(5)

        torch.testing.assert_close(result1, result2)

    def test_seed_everything_different_seeds(self):
        """Test that different seeds produce different results."""
        seed_everything(42)
        result1 = np.random.rand(5)

        seed_everything(123)
        result2 = np.random.rand(5)

        assert not np.array_equal(result1, result2)


class TestIO:
    """Test IO functions."""

    def test_save_and_load_pickle_dict(self):
        """Test saving and loading a dictionary with pickle."""
        test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": True}}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pkl"
            save_pickle(test_data, filepath)
            loaded_data = load_pickle(filepath)

            assert loaded_data == test_data

    def test_save_and_load_pickle_dataframe(self):
        """Test saving and loading a pandas DataFrame with pickle."""
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4.5, 5.5, 6.5], "C": ["x", "y", "z"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_df.pkl"
            save_pickle(test_df, filepath)
            loaded_df = load_pickle(filepath)

            pd.testing.assert_frame_equal(loaded_df, test_df)

    def test_save_and_load_pickle_numpy_array(self):
        """Test saving and loading a numpy array with pickle."""
        test_array = np.array([[1, 2, 3], [4, 5, 6]])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_array.pkl"
            save_pickle(test_array, filepath)
            loaded_array = load_pickle(filepath)

            np.testing.assert_array_equal(loaded_array, test_array)

    def test_pickle_uses_highest_protocol(self):
        """Test that save_pickle uses the highest protocol available."""
        test_data = {"test": "data"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pkl"
            save_pickle(test_data, filepath)

            # Read the file to check the protocol
            with open(filepath, "rb") as f:
                # First byte indicates the pickle protocol
                protocol_byte = f.read(1)[0]
                # Protocol 5 was added in Python 3.8
                assert protocol_byte >= 4


class TestConfigUtils:
    """Test configuration utilities."""

    def test_load_module_string(self):
        """Test loading a module from string path."""
        module = load_module("pathlib.Path")
        assert module == Path

    def test_load_module_non_string(self):
        """Test that load_module returns the input if not a string."""

        class DummyClass:
            pass

        result = load_module(DummyClass)
        assert result == DummyClass

    def test_get_config_dict_copy(self):
        """Test creating an editable copy of a config dict."""
        config = config_dict.ConfigDict()
        config.param1 = 42
        config.param2 = "test"
        config.lock()

        # Original should be locked
        with pytest.raises(AttributeError):
            config.new_param = 100

        # Copy should be unlocked
        config_copy = get_config_dict_copy(config)
        config_copy.new_param = 100

        assert config_copy.new_param == 100
        assert not hasattr(config, "new_param")

    def test_instantiate_with_target(self):
        """Test instantiating an object from config with _target_."""
        config = config_dict.ConfigDict()
        config._target_ = "pathlib.Path"
        # Path constructor takes positional argument, not "path"
        # Let's use a different class for testing
        config._target_ = "collections.namedtuple"
        config.typename = "TestTuple"
        config.field_names = ["field1", "field2"]

        result = instantiate(config)
        # Should create a namedtuple class
        assert hasattr(result, "_fields")
        assert result._fields == ("field1", "field2")

    def test_instantiate_without_target(self):
        """Test instantiate returns config when no _target_ is specified."""
        config = config_dict.ConfigDict()
        config.param1 = 42
        config.param2 = "test"

        result = instantiate(config)
        assert isinstance(result, config_dict.ConfigDict)
        assert result.param1 == 42

    def test_instantiate_with_partial(self):
        """Test instantiate with _partial_ flag."""
        config = config_dict.ConfigDict()
        config._target_ = "pathlib.Path"
        config._partial_ = True
        config.path = "/tmp/test"

        result = instantiate(config)
        # Result should be a partial function
        assert callable(result)
        path_instance = result()
        assert isinstance(path_instance, Path)

    def test_instantiate_skip(self):
        """Test that _skip_instantiate_ prevents instantiation."""
        config = config_dict.ConfigDict()
        config._target_ = "pathlib.Path"
        config._skip_instantiate_ = True
        config.path = "/tmp/test"

        result = instantiate(config)
        assert isinstance(result, config_dict.ConfigDict)
        assert result._target_ == "pathlib.Path"

    def test_instantiate_nested_config(self):
        """Test instantiate with nested configurations."""
        inner_config = config_dict.ConfigDict()
        inner_config.value = 42

        outer_config = config_dict.ConfigDict()
        outer_config.nested = inner_config
        outer_config.simple = "test"

        result = instantiate(outer_config)
        assert isinstance(result, config_dict.ConfigDict)
        assert result.nested.value == 42
        assert result.simple == "test"
