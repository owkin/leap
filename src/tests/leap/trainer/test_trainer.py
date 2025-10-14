"""Tests for the trainer module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_collections import config_dict

from leap.trainer.perturbation_model_trainer import (
    PerturbationModelTrainer,
    SplitIds,
    SplitPairIds,
    check_list_pair,
)


class TestCheckListPair:
    """Test check_list_pair function."""

    def test_valid_list_of_pairs(self):
        """Test with valid list of pairs."""
        pairs = [("a", 1), ("b", 2), ("c", 3)]
        result = check_list_pair(pairs)
        assert result == pairs

    def test_invalid_not_list(self):
        """Test with non-list input."""
        with pytest.raises(ValueError, match="Expected a list"):
            check_list_pair("not a list")

    def test_invalid_not_tuples(self):
        """Test with list not containing tuples."""
        with pytest.raises(ValueError, match="Expected a list of pairs"):
            check_list_pair([1, 2, 3])

    def test_invalid_tuple_length(self):
        """Test with tuples of wrong length."""
        with pytest.raises(ValueError, match="Expected a list of pairs"):
            check_list_pair([("a", 1, "extra"), ("b", 2)])


class TestPerturbationModelTrainer:
    """Test PerturbationModelTrainer class."""

    @pytest.fixture
    def toy_config(self):
        """Create toy configuration for testing."""
        # Source domain data config
        source_config = config_dict.ConfigDict()
        source_config._target_ = "leap.data.preclinical_dataset.PreclinicalDataset"
        source_config.label = None  # No label for simplicity
        source_config.min_n_label = 0

        # Data split config
        split_config = config_dict.ConfigDict()
        split_config._target_ = "sklearn.model_selection.KFold"
        split_config.n_splits = 2
        split_config.shuffle = True
        split_config.random_state = 42

        # Model config
        model_config = config_dict.ConfigDict()
        model_config._target_ = "leap.pipelines.perturbation_pipeline.PerturbationPipeline"
        model_config.preprocessor_model_rnaseq = None
        model_config.rpz_model_rnaseq = None
        regression_model = config_dict.ConfigDict()
        regression_model._target_ = "leap.regression_models.ElasticNet"
        regression_model.alpha = 0.1
        model_config.regression_model_base_instance = regression_model
        model_config.hpt_tuning_cv_split = None
        model_config.hpt_tuning_param_grid = None
        model_config.hpt_tuning_score = None
        model_config.fgpt_rpz_model = None
        model_config.one_model_per_perturbation = True
        model_config.ensembling = False

        return source_config, split_config, model_config

    @pytest.fixture
    def toy_dataset(self):
        """Create toy dataset for testing."""
        np.random.seed(42)
        n_samples = 20
        n_genes = 10
        n_perturbations = 3

        # Create a mock dataset
        class MockDataset:
            def __init__(self):
                self.df_rnaseq = pd.DataFrame(
                    np.random.randn(n_samples, n_genes),
                    columns=[f"gene_{i}_rnaseq" for i in range(n_genes)],
                    index=[f"sample_{i}" for i in range(n_samples)],
                )

                self.df_labels = pd.DataFrame(
                    np.random.randn(n_samples, n_perturbations),
                    columns=[f"pert_{i}" for i in range(n_perturbations)],
                    index=[f"sample_{i}" for i in range(n_samples)],
                )

                self.df_sample_metadata = pd.DataFrame(
                    {"tissue": ["Lung"] * n_samples, "domain": ["source"] * n_samples},
                    index=[f"sample_{i}" for i in range(n_samples)],
                )

                self.df_fingerprints = None

                # Create stacked versions
                self.df_labels_stacked = self.df_labels.stack()
                self.df_labels_stacked.index.names = ["sample", "perturbation"]
                self.df_labels_stacked = pd.DataFrame(self.df_labels_stacked, columns=["label"])

                self.df_sample_metadata_stacked = pd.merge(
                    self.df_sample_metadata, self.df_labels_stacked, left_index=True, right_on="sample"
                )

        return MockDataset()

    def test_trainer_initialization(self, toy_config):
        """Test trainer initialization."""
        source_config, split_config, model_config = toy_config

        trainer = PerturbationModelTrainer(
            source_domain_data=source_config, data_split=split_config, model=model_config
        )

        assert trainer.config_source_domain_data is not None
        assert trainer.config_data_split is not None
        assert trainer.config_model is not None

    def test_trainer_extract_sample_ids(self):
        """Test extracting sample IDs from pair IDs."""
        trainer = PerturbationModelTrainer(
            source_domain_data=config_dict.ConfigDict(),
            data_split=config_dict.ConfigDict(),
            model=config_dict.ConfigDict(),
        )

        split_pair_ids = {
            "split_0": SplitPairIds(
                training_ids=[("sample_0", "pert_0"), ("sample_1", "pert_0")], test_ids=[("sample_2", "pert_0")]
            )
        }

        sample_ids = trainer.extract_split_sample_ids(split_pair_ids)

        assert "split_0" in sample_ids
        assert set(sample_ids["split_0"]["training_ids"]) == {"sample_0", "sample_1"}
        assert set(sample_ids["split_0"]["test_ids"]) == {"sample_2"}

    def test_trainer_extract_perturbation_ids(self):
        """Test extracting perturbation IDs from pair IDs."""
        trainer = PerturbationModelTrainer(
            source_domain_data=config_dict.ConfigDict(),
            data_split=config_dict.ConfigDict(),
            model=config_dict.ConfigDict(),
        )

        split_pair_ids = {
            "split_0": SplitPairIds(
                training_ids=[("sample_0", "pert_0"), ("sample_0", "pert_1")], test_ids=[("sample_1", "pert_0")]
            )
        }

        pert_ids = trainer.extract_split_perturbation_ids(split_pair_ids)

        assert "split_0" in pert_ids
        assert set(pert_ids["split_0"]["training_ids"]) == {"pert_0", "pert_1"}
        assert set(pert_ids["split_0"]["test_ids"]) == {"pert_0"}

    def test_trainer_get_pair_member(self):
        """Test _get_pair_member static method."""
        pairs = [("a", 1), ("b", 2), ("c", 3)]

        # Get first elements
        first_elements = PerturbationModelTrainer._get_pair_member(pairs, 0)
        assert set(first_elements) == {"a", "b", "c"}

        # Get second elements
        second_elements = PerturbationModelTrainer._get_pair_member(pairs, 1)
        assert set(second_elements) == {1, 2, 3}

    def test_trainer_get_pair_member_invalid_id(self):
        """Test _get_pair_member with invalid pair_id."""
        pairs = [("a", 1), ("b", 2)]

        with pytest.raises(ValueError, match="pair_id must be 0 or 1"):
            PerturbationModelTrainer._get_pair_member(pairs, 2)

    def test_trainer_aggregate_performances(self):
        """Test aggregate_performances static method."""
        test_performance = {
            "overall": {"spearman": {"split_0": {"overall": 0.8}, "split_1": {"overall": 0.85}}},
            "per_perturbation": {
                "spearman": {"split_0": {"pert_0": 0.7, "pert_1": 0.9}, "split_1": {"pert_0": 0.75, "pert_1": 0.95}}
            },
        }

        aggregated = PerturbationModelTrainer.aggregate_performances(test_performance, format_numbers=True)

        assert "overall" in aggregated
        assert "per_perturbation" in aggregated
        assert "spearman" in aggregated["overall"]
        assert "mean" in aggregated["overall"]["spearman"]
        assert "std" in aggregated["overall"]["spearman"]

    def test_trainer_aggregate_performances_no_format(self):
        """Test aggregate without formatting."""
        test_performance = {"overall": {"spearman": {"split_0": {"overall": 0.8}, "split_1": {"overall": 0.85}}}}

        aggregated = PerturbationModelTrainer.aggregate_performances(test_performance, format_numbers=False)

        # Values should be floats, not strings
        assert isinstance(aggregated["overall"]["spearman"]["mean"], (float, np.floating))

    def test_trainer_keep_n_splits(self):
        """Test _keep_n_splits method."""
        trainer = PerturbationModelTrainer(
            source_domain_data=config_dict.ConfigDict(),
            data_split=config_dict.ConfigDict(),
            model=config_dict.ConfigDict(),
        )

        split_pair_ids = {
            "split_0": SplitPairIds(training_ids=[], test_ids=[]),
            "split_1": SplitPairIds(training_ids=[], test_ids=[]),
            "split_2": SplitPairIds(training_ids=[], test_ids=[]),
            "split_3": SplitPairIds(training_ids=[], test_ids=[]),
        }

        # Keep 2 splits starting from split 1
        result = trainer._keep_n_splits(split_pair_ids, start_split_n=1, n_splits=2)

        assert len(result) == 2
        assert "split_1" in result
        assert "split_2" in result

    def test_trainer_keep_n_splits_invalid_start(self):
        """Test _keep_n_splits with invalid start."""
        trainer = PerturbationModelTrainer(
            source_domain_data=config_dict.ConfigDict(),
            data_split=config_dict.ConfigDict(),
            model=config_dict.ConfigDict(),
        )

        split_pair_ids = {
            "split_0": SplitPairIds(training_ids=[], test_ids=[]),
        }

        with pytest.raises(ValueError, match="Not enough splits to keep"):
            trainer._keep_n_splits(split_pair_ids, start_split_n=5, n_splits=1)

    def test_trainer_keep_n_splits_not_enough_splits(self):
        """Test _keep_n_splits when not enough splits."""
        trainer = PerturbationModelTrainer(
            source_domain_data=config_dict.ConfigDict(),
            data_split=config_dict.ConfigDict(),
            model=config_dict.ConfigDict(),
        )

        split_pair_ids = {
            "split_0": SplitPairIds(training_ids=[], test_ids=[]),
        }

        with pytest.raises(ValueError, match="Not enough splits"):
            trainer._keep_n_splits(split_pair_ids, start_split_n=0, n_splits=5)

    def test_trainer_str_method(self, toy_config):
        """Test __str__ method."""
        source_config, split_config, model_config = toy_config

        trainer = PerturbationModelTrainer(
            source_domain_data=source_config, data_split=split_config, model=model_config
        )

        str_repr = str(trainer)
        assert "config_source_domain_data" in str_repr
        assert "config_data_split" in str_repr
        assert "config_model" in str_repr

    def test_trainer_output_path_property(self, toy_config):
        """Test output_path property."""
        source_config, split_config, model_config = toy_config

        trainer = PerturbationModelTrainer(
            source_domain_data=source_config, data_split=split_config, model=model_config
        )

        # Should raise error when not set
        with pytest.raises(ValueError, match="Output path is not set"):
            _ = trainer.output_path

        # Should work after setting
        trainer.output_path = Path("/tmp/test")
        assert trainer.output_path == Path("/tmp/test")

    def test_trainer_data_property(self, toy_config, toy_dataset):
        """Test data property."""
        source_config, split_config, model_config = toy_config

        trainer = PerturbationModelTrainer(
            source_domain_data=source_config, data_split=split_config, model=model_config
        )

        # Should raise error when not set
        with pytest.raises(ValueError, match="Data is not loaded"):
            _ = trainer.data

        # Should work after setting
        trainer.data = toy_dataset
        assert trainer.data is not None

    def test_trainer_compute_performances(self, toy_config):
        """Test compute_performances method."""
        source_config, split_config, model_config = toy_config

        trainer = PerturbationModelTrainer(
            source_domain_data=source_config, data_split=split_config, model=model_config
        )

        # Create toy predictions and true labels
        test_true_labels = {"split_0": pd.DataFrame({"pert_0": [1.0, 2.0, 3.0], "pert_1": [4.0, 5.0, 6.0]})}

        test_predicted_labels = {"split_0": pd.DataFrame({"pert_0": [1.1, 2.1, 3.1], "pert_1": [4.1, 5.1, 6.1]})}

        performances = trainer.compute_performances(
            test_true_labels=test_true_labels,
            test_predicted_labels=test_predicted_labels,
            performance_per_perturbation=[True, False],
            metric=["spearman", "pearson"],
        )

        assert "overall" in performances
        assert "per_perturbation" in performances
        assert "spearman" in performances["overall"]
        assert "pearson" in performances["overall"]

    def test_trainer_log_data_summary(self, toy_config, toy_dataset):
        """Test log_data_summary method."""
        source_config, split_config, model_config = toy_config

        trainer = PerturbationModelTrainer(
            source_domain_data=source_config, data_split=split_config, model=model_config
        )

        # Should not raise error
        trainer.log_data_summary(data=toy_dataset)

    def test_trainer_save_data_summary(self, toy_config, toy_dataset):
        """Test save_data_summary method."""
        source_config, split_config, model_config = toy_config

        trainer = PerturbationModelTrainer(
            source_domain_data=source_config, data_split=split_config, model=model_config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "data_summary.csv"
            trainer.save_data_summary(data=toy_dataset, output_path=output_path)

            # File should exist
            assert output_path.exists()

            # Should be readable
            df = pd.read_csv(output_path)
            assert len(df) > 0


class TestSplitTypes:
    """Test SplitPairIds and SplitIds types."""

    def test_split_pair_ids_creation(self):
        """Test SplitPairIds type."""
        split_pair = SplitPairIds(training_ids=[("sample_0", "pert_0")], test_ids=[("sample_1", "pert_1")])

        assert len(split_pair["training_ids"]) == 1
        assert len(split_pair["test_ids"]) == 1

    def test_split_ids_creation(self):
        """Test SplitIds type."""
        split_id = SplitIds(training_ids=["sample_0", "sample_1"], test_ids=["sample_2"])

        assert len(split_id["training_ids"]) == 2
        assert len(split_id["test_ids"]) == 1
