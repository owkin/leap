"""Tests for the splits module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leap.data.splits import (
    _group_shuffle_split_by_count,
    _leave_n_samples_out_split_generator,
    _validate_cv_split_params,
    cv_split_generator,
    cv_split_ids,
    leave_one_group_out_split_generator,
)


class TestValidateCvSplitParams:
    """Test _validate_cv_split_params function."""

    def test_leave_one_group_out_without_group_variable(self):
        """Test that leave_one_group_out requires group_variable."""
        with pytest.raises(ValueError, match="Cannot leave one group out without a grouping variable"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable=None,
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=True,
                test_split_ratio=None,
                training_split_count=None,
                n_splits=None,
            )

    def test_leave_one_group_out_with_stratify(self):
        """Test that leave_one_group_out cannot be combined with stratify."""
        with pytest.raises(NotImplementedError, match="Cannot stratify and leave one group out"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable="tissue",
                subgroup_variable=None,
                stratify_variable="label",
                leave_one_group_out=True,
                test_split_ratio=None,
                training_split_count=None,
                n_splits=None,
            )

    def test_leave_one_group_out_with_subgroup(self):
        """Test that leave_one_group_out cannot be combined with subgroup."""
        with pytest.raises(NotImplementedError, match="Cannot split by subgroup and leave one group out"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable="tissue",
                subgroup_variable="perturbation",
                stratify_variable=None,
                leave_one_group_out=True,
                test_split_ratio=None,
                training_split_count=None,
                n_splits=None,
            )

    def test_leave_one_group_out_with_n_splits(self):
        """Test that leave_one_group_out cannot fix n_splits."""
        with pytest.raises(ValueError, match="Cannot fix the number of splits with leave one group out"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable="tissue",
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=True,
                test_split_ratio=None,
                training_split_count=None,
                n_splits=5,
            )

    def test_leave_one_group_out_with_test_split_ratio(self):
        """Test that leave_one_group_out cannot fix test_split_ratio."""
        with pytest.raises(ValueError, match="Cannot fix the test split ratio with leave one group out"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable="tissue",
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=True,
                test_split_ratio=0.2,
                training_split_count=None,
                n_splits=None,
            )

    def test_leave_one_group_out_with_training_split_count(self):
        """Test that leave_one_group_out cannot fix training_split_count."""
        with pytest.raises(ValueError, match="Cannot fix the number of training samples with leave one group out"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable="tissue",
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=True,
                test_split_ratio=None,
                training_split_count=100,
                n_splits=None,
            )

    def test_leave_one_group_out_with_k_fold(self):
        """Test that leave_one_group_out cannot be combined with k_fold."""
        with pytest.raises(ValueError, match="Cannot do K fold and leave one group out"):
            _validate_cv_split_params(
                k_fold=True,
                group_variable="tissue",
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=True,
                test_split_ratio=None,
                training_split_count=None,
                n_splits=None,
            )

    def test_k_fold_with_test_split_ratio(self):
        """Test that k_fold cannot fix test_split_ratio."""
        with pytest.raises(ValueError, match="Cannot fix the test split ratio with K folds"):
            _validate_cv_split_params(
                k_fold=True,
                group_variable=None,
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=False,
                test_split_ratio=0.2,
                training_split_count=None,
                n_splits=5,
            )

    def test_test_split_ratio_too_small(self):
        """Test that test_split_ratio must be > 0."""
        with pytest.raises(ValueError, match="Test split ratio must be > 0"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable=None,
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=False,
                test_split_ratio=0.0,
                training_split_count=None,
                n_splits=5,
            )

    def test_test_split_ratio_too_large(self):
        """Test that test_split_ratio must be < 1."""
        with pytest.raises(ValueError, match="Test split ratio must be < 1"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable=None,
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=False,
                test_split_ratio=1.0,
                training_split_count=None,
                n_splits=5,
            )

    def test_training_split_count_with_test_split_ratio(self):
        """Test that training_split_count and test_split_ratio cannot both be provided."""
        with pytest.raises(ValueError, match="Either test_split_ratio or training_split_count should be provided"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable=None,
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=False,
                test_split_ratio=0.2,
                training_split_count=100,
                n_splits=5,
            )

    def test_training_split_count_with_k_fold(self):
        """Test that training_split_count cannot be used with k_fold."""
        with pytest.raises(ValueError, match="Cannot use k_fold if training_split_count is provided"):
            _validate_cv_split_params(
                k_fold=True,
                group_variable=None,
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=False,
                test_split_ratio=None,
                training_split_count=100,
                n_splits=5,
            )

    def test_training_split_count_with_leave_one_group_out(self):
        """Test that training_split_count cannot be used with leave_one_group_out."""
        with pytest.raises(ValueError, match="Cannot fix the number of training samples with leave one group out"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable="tissue",
                subgroup_variable=None,
                stratify_variable=None,
                leave_one_group_out=True,
                test_split_ratio=None,
                training_split_count=100,
                n_splits=None,
            )

    def test_subgroup_variable_without_training_split_count(self):
        """Test that subgroup_variable requires training_split_count."""
        with pytest.raises(ValueError, match="training_split_count must be provided when using subgroup_variable"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable=None,
                subgroup_variable="perturbation",
                stratify_variable=None,
                leave_one_group_out=False,
                test_split_ratio=0.2,
                training_split_count=None,
                n_splits=5,
            )

    def test_group_and_stratify_with_test_split_ratio(self):
        """Test that group and stratify cannot be combined with test_split_ratio."""
        with pytest.raises(NotImplementedError, match="Cannot stratify and group at the same time"):
            _validate_cv_split_params(
                k_fold=False,
                group_variable="tissue",
                subgroup_variable=None,
                stratify_variable="label",
                leave_one_group_out=False,
                test_split_ratio=0.2,
                training_split_count=None,
                n_splits=5,
            )

    def test_valid_params(self):
        """Test that valid parameters don't raise errors."""
        # Should not raise
        _validate_cv_split_params(
            k_fold=False,
            group_variable=None,
            subgroup_variable=None,
            stratify_variable=None,
            leave_one_group_out=False,
            test_split_ratio=0.2,
            training_split_count=None,
            n_splits=5,
        )


class TestCvSplitIds:
    """Test cv_split_ids function."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "tissue": np.random.choice(["Lung", "Breast", "Liver"], n_samples),
                "perturbation": np.random.choice(["pert_A", "pert_B", "pert_C"], n_samples),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

    def test_shuffle_split(self, sample_metadata):
        """Test basic shuffle split."""
        split_ids = cv_split_ids(
            X_metadata=sample_metadata,
            k_fold=False,
            test_split_ratio=0.2,
            n_splits=3,
            random_state=42,
        )

        assert len(split_ids) == 3
        assert "split_0" in split_ids
        assert "training_ids" in split_ids["split_0"]
        assert "test_ids" in split_ids["split_0"]

        # Check approximate split ratio
        total = len(sample_metadata)
        test_size = len(split_ids["split_0"]["test_ids"])
        assert 0.15 < test_size / total < 0.25

    def test_k_fold_split(self, sample_metadata):
        """Test K-fold split."""
        split_ids = cv_split_ids(
            X_metadata=sample_metadata,
            k_fold=True,
            test_split_ratio=None,  # Must be None for k_fold
            n_splits=5,
            random_state=42,
        )

        assert len(split_ids) == 5

        # Check that splits are roughly equal
        test_sizes = [len(split_ids[f"split_{i}"]["test_ids"]) for i in range(5)]
        assert all(15 <= size <= 25 for size in test_sizes)

    def test_group_shuffle_split(self, sample_metadata):
        """Test group shuffle split."""
        split_ids = cv_split_ids(
            X_metadata=sample_metadata,
            k_fold=False,
            group_variable="tissue",
            test_split_ratio=0.2,
            n_splits=3,
            random_state=42,
        )

        assert len(split_ids) == 3

        # Verify that groups are not split across train/test
        for split_name in split_ids:
            train_ids = split_ids[split_name]["training_ids"]
            test_ids = split_ids[split_name]["test_ids"]

            train_tissues = set(sample_metadata.loc[train_ids, "tissue"])
            test_tissues = set(sample_metadata.loc[test_ids, "tissue"])

            # No tissue should appear in both train and test
            assert len(train_tissues.intersection(test_tissues)) == 0

    def test_stratified_shuffle_split(self, sample_metadata):
        """Test stratified shuffle split."""
        # Add a binary label for stratification
        sample_metadata["label"] = np.random.choice([0, 1], len(sample_metadata))

        split_ids = cv_split_ids(
            X_metadata=sample_metadata,
            k_fold=False,
            stratify_variable="label",
            test_split_ratio=0.2,
            n_splits=3,
            random_state=42,
        )

        assert len(split_ids) == 3

        # Check that stratification is roughly preserved
        overall_ratio = sample_metadata["label"].mean()

        for split_name in split_ids:
            test_ids = split_ids[split_name]["test_ids"]
            test_ratio = sample_metadata.loc[test_ids, "label"].mean()

            # Should be within 0.1 of overall ratio
            assert abs(test_ratio - overall_ratio) < 0.15

    def test_test_split_ratio_zero(self, sample_metadata):
        """Test with test_split_ratio=0 (all training)."""
        split_ids = cv_split_ids(
            X_metadata=sample_metadata,
            test_split_ratio=0,
            n_splits=1,
            random_state=42,
        )

        assert len(split_ids) == 1
        assert len(split_ids["split_0"]["training_ids"]) == len(sample_metadata)
        assert len(split_ids["split_0"]["test_ids"]) == 0

    def test_test_split_ratio_zero_with_multiple_splits_raises(self, sample_metadata):
        """Test that test_split_ratio=0 with multiple splits raises error."""
        with pytest.raises(ValueError, match="Cannot have more than one split with test_split_ratio = 0"):
            cv_split_ids(
                X_metadata=sample_metadata,
                test_split_ratio=0,
                n_splits=3,
                random_state=42,
            )

    def test_test_split_ratio_one(self, sample_metadata):
        """Test with test_split_ratio=1 (all test)."""
        split_ids = cv_split_ids(
            X_metadata=sample_metadata,
            test_split_ratio=1,
            n_splits=1,
            random_state=42,
        )

        assert len(split_ids) == 1
        assert len(split_ids["split_0"]["training_ids"]) == 0
        assert len(split_ids["split_0"]["test_ids"]) == len(sample_metadata)

    def test_test_split_ratio_one_with_multiple_splits_raises(self, sample_metadata):
        """Test that test_split_ratio=1 with multiple splits raises error."""
        with pytest.raises(ValueError, match="Cannot have more than one split with test_split_ratio = 1"):
            cv_split_ids(
                X_metadata=sample_metadata,
                test_split_ratio=1,
                n_splits=3,
                random_state=42,
            )

    def test_leave_one_group_out_split(self, sample_metadata):
        """Test leave-one-group-out split."""
        split_ids = cv_split_ids(
            X_metadata=sample_metadata,
            k_fold=False,
            group_variable="tissue",
            leave_one_group_out=True,
            test_split_ratio=None,  # Must be None for leave_one_group_out
            n_splits=None,  # Must be None for leave_one_group_out
            n_min_loo=5,
            random_state=42,
        )

        # Should have one split per tissue (if each has enough samples)
        assert len(split_ids) >= 1

        # Check that each split uses one group as test
        for split_name in split_ids:
            if "ood_test" in split_ids[split_name]:
                test_group = split_ids[split_name]["ood_test"]
                test_ids = split_ids[split_name]["test_ids"]

                # All test samples should be from the test group
                test_tissues = sample_metadata.loc[test_ids, "tissue"].unique()
                assert len(test_tissues) == 1
                assert test_tissues[0] == test_group


class TestLeaveOneGroupOutSplitGenerator:
    """Test leave_one_group_out_split_generator function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 120
        return pd.DataFrame(
            {
                "tissue": np.repeat(["Lung", "Breast", "Liver", "Brain"], 30),
                "perturbation": np.tile(["pert_A", "pert_B", "pert_C"], 40),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

    def test_leave_one_group_out_with_return_group(self, sample_data):
        """Test leave-one-group-out with return_group=True."""
        generator = leave_one_group_out_split_generator(
            df_all=sample_data,
            group_variable="tissue",
            n_min_loo=10,
            return_group=True,
        )

        splits = list(generator)

        # Should have 4 splits (one per tissue)
        assert len(splits) == 4

        # Each split should return (train_ids, test_ids, group)
        for train_ids, test_ids, group in splits:
            assert isinstance(train_ids, list)
            assert isinstance(test_ids, list)
            assert isinstance(group, str)
            assert group in ["Lung", "Breast", "Liver", "Brain"]

            # Test set should be from the held-out group
            test_tissues = sample_data.iloc[test_ids]["tissue"].unique()
            assert len(test_tissues) == 1
            assert test_tissues[0] == group

    def test_leave_one_group_out_without_return_group(self, sample_data):
        """Test leave-one-group-out with return_group=False."""
        generator = leave_one_group_out_split_generator(
            df_all=sample_data,
            group_variable="tissue",
            n_min_loo=10,
            return_group=False,
        )

        splits = list(generator)

        # Should have 4 splits
        assert len(splits) == 4

        # Each split should return (train_ids, test_ids) only
        for split in splits:
            assert len(split) == 2
            train_ids, test_ids = split
            assert isinstance(train_ids, list)
            assert isinstance(test_ids, list)

    def test_leave_one_group_out_with_min_samples(self, sample_data):
        """Test that groups with too few samples are filtered out."""
        # Create data where one group has very few samples
        small_data = pd.DataFrame(
            {
                "tissue": ["Lung"] * 50 + ["Breast"] * 5,
                "perturbation": ["pert_A"] * 55,
            }
        )

        generator = leave_one_group_out_split_generator(
            df_all=small_data,
            group_variable="tissue",
            n_min_loo=10,
            return_group=True,
        )

        splits = list(generator)

        # Only Lung should be included (Breast has < 10 samples)
        assert len(splits) == 1
        _, _, group = splits[0]
        assert group == "Lung"

    def test_leave_one_group_out_with_list_test_groups(self, sample_data):
        """Test leave-one-group-out with specific test groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with specific test groups
            test_groups_file = Path(tmpdir) / "test_groups.csv"
            pd.DataFrame(["Lung", "Liver"]).to_csv(test_groups_file, index=False, header=False)

            generator = leave_one_group_out_split_generator(
                df_all=sample_data,
                group_variable="tissue",
                n_min_loo=10,
                list_test_groups=test_groups_file,
                return_group=True,
            )

            splits = list(generator)

            # Should only have splits for Lung and Liver
            groups = [group for _, _, group in splits]
            assert set(groups) == {"Lung", "Liver"}


class TestLeaveNSamplesOutSplitGenerator:
    """Test _leave_n_samples_out_split_generator function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 150
        return pd.DataFrame(
            {
                "perturbation": np.repeat(["pert_A", "pert_B", "pert_C"], 50),
                "tissue": np.random.choice(["Lung", "Breast"], n_samples),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

    def test_leave_n_samples_out(self, sample_data):
        """Test leave n samples out split generator."""
        generator = _leave_n_samples_out_split_generator(
            X_metadata=sample_data,
            subgroup_variable="perturbation",
            training_split_count=10,
            n_splits=3,
            random_state=42,
        )

        splits = list(generator)

        assert len(splits) == 3

        for train_ids, test_ids in splits:
            # Each perturbation should have approximately 10 training samples
            train_data = sample_data.iloc[train_ids]
            train_counts = train_data["perturbation"].value_counts()

            # Should have exactly 10 per perturbation (3 perturbations * 10 = 30 total)
            assert len(train_ids) == 30
            assert all(count == 10 for count in train_counts)

    def test_leave_n_samples_out_with_stratify(self, sample_data):
        """Test leave n samples out with stratification."""
        generator = _leave_n_samples_out_split_generator(
            X_metadata=sample_data,
            subgroup_variable="perturbation",
            training_split_count=10,
            n_splits=2,
            stratify_variable="tissue",
            random_state=42,
        )

        splits = list(generator)

        assert len(splits) == 2

        for train_ids, test_ids in splits:
            assert len(train_ids) > 0
            assert len(test_ids) > 0


class TestGroupShuffleSplitByCount:
    """Test _group_shuffle_split_by_count function."""

    def test_group_shuffle_split_basic(self):
        """Test basic group shuffle split by count."""
        df = pd.DataFrame(
            {
                "value": range(50),
            },
            index=range(50),
        )

        result = _group_shuffle_split_by_count(
            df_by_stratum=df,
            training_split_count=10,
            random_state=42,
        )

        assert "training_ids" in result
        assert "test_ids" in result
        assert len(result["training_ids"]) == 10
        assert len(result["test_ids"]) == 40

    def test_group_shuffle_split_with_stratify(self):
        """Test group shuffle split with stratification."""
        df = pd.DataFrame(
            {
                "value": range(50),
                "label": [0] * 25 + [1] * 25,
            },
            index=range(50),
        )

        result = _group_shuffle_split_by_count(
            df_by_stratum=df,
            training_split_count=10,
            stratify_variable="label",
            random_state=42,
        )

        # With stratify, the logic creates train split with (n_total - training_split_count) samples
        # So with training_split_count=10 and 50 total, we get 40 training and 10 test
        assert len(result["training_ids"]) == 40
        assert len(result["test_ids"]) == 10


class TestCvSplitGenerator:
    """Test cv_split_generator function."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "tissue": np.random.choice(["Lung", "Breast", "Liver"], n_samples),
                "label": np.random.choice([0, 1], n_samples),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

    def test_shuffle_split_generator(self, sample_metadata):
        """Test shuffle split generator."""
        generator = cv_split_generator(
            X_metadata=sample_metadata,
            k_fold=False,
            test_split_ratio=0.2,
            n_splits=3,
            random_state=42,
        )

        splits = list(generator)
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx).intersection(set(test_idx))) == 0

    def test_k_fold_generator(self, sample_metadata):
        """Test K-fold generator."""
        generator = cv_split_generator(
            X_metadata=sample_metadata,
            k_fold=True,
            test_split_ratio=None,  # Must be None for k_fold
            n_splits=5,
            random_state=42,
        )

        splits = list(generator)
        assert len(splits) == 5

    def test_group_k_fold_generator(self, sample_metadata):
        """Test group K-fold generator."""
        generator = cv_split_generator(
            X_metadata=sample_metadata,
            k_fold=True,
            group_variable="tissue",
            test_split_ratio=None,  # Must be None for k_fold
            n_splits=3,
            random_state=42,
        )

        splits = list(generator)
        assert len(splits) == 3

    def test_stratified_k_fold_generator(self, sample_metadata):
        """Test stratified K-fold generator."""
        generator = cv_split_generator(
            X_metadata=sample_metadata,
            k_fold=True,
            stratify_variable="label",
            test_split_ratio=None,  # Must be None for k_fold
            n_splits=5,
            random_state=42,
        )

        splits = list(generator)
        assert len(splits) == 5
