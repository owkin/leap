"""Tests for the data module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leap.data.preprocessor import SCALERS, OmicsPreprocessor


class TestPreprocessor:
    """Test OmicsPreprocessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        # Create data with 50 samples and 100 genes
        data = pd.DataFrame(np.random.exponential(scale=10, size=(50, 100)), columns=[f"gene_{i}" for i in range(100)])
        return data

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = OmicsPreprocessor(scaling_method="min_max", max_genes=50, log_scaling=True)

        assert preprocessor.scaling_method == "min_max"
        assert preprocessor.max_genes == 50
        assert preprocessor.log_scaling is True

    def test_preprocessor_invalid_scaler_raises_error(self):
        """Test that invalid scaler raises ValueError."""
        with pytest.raises(ValueError, match="Scaling method must be"):
            OmicsPreprocessor(scaling_method="invalid_scaler")

    def test_preprocessor_fit_transform_min_max(self, sample_data):
        """Test fit_transform with min_max scaling."""
        preprocessor = OmicsPreprocessor(scaling_method="min_max", max_genes=-1, log_scaling=False)

        transformed = preprocessor.fit_transform(sample_data)

        # Check that data is scaled between 0 and 1 (with tolerance for floating point)
        assert transformed.min().min() >= -1e-10
        assert transformed.max().max() <= 1 + 1e-10
        assert transformed.shape == sample_data.shape

    def test_preprocessor_fit_transform_mean_std(self, sample_data):
        """Test fit_transform with mean_std scaling."""
        preprocessor = OmicsPreprocessor(scaling_method="mean_std", max_genes=-1, log_scaling=False)

        transformed = preprocessor.fit_transform(sample_data)

        # Check that data is standardized (mean ~ 0, std ~ 1)
        assert abs(transformed.mean().mean()) < 0.1
        assert abs(transformed.std().mean() - 1.0) < 0.1

    def test_preprocessor_log_scaling(self, sample_data):
        """Test log scaling."""
        preprocessor = OmicsPreprocessor(scaling_method="identity", max_genes=-1, log_scaling=True)

        transformed = preprocessor.fit_transform(sample_data)

        # Check that log transform was applied
        # Note: columns are sorted by preprocessor
        expected = np.log1p(sample_data).sort_index(axis=1)
        pd.testing.assert_frame_equal(transformed, expected, check_dtype=False)

    def test_preprocessor_gene_selection(self, sample_data):
        """Test gene selection based on variance."""
        max_genes = 20
        preprocessor = OmicsPreprocessor(scaling_method="identity", max_genes=max_genes, log_scaling=False)

        transformed = preprocessor.fit_transform(sample_data)

        # Should keep only max_genes
        assert transformed.shape[1] == max_genes
        assert len(preprocessor.columns_to_keep) == max_genes

    def test_preprocessor_gene_list_source_file(self, sample_data):
        """Test gene selection from file."""
        # Preprocessor reads first column and intersects with available genes
        selected_genes = ["gene_0", "gene_1", "gene_2", "gene_3"]

        with tempfile.TemporaryDirectory() as tmpdir:
            gene_list_file = Path(tmpdir) / "genes.csv"
            pd.DataFrame(selected_genes).to_csv(gene_list_file, index=False, header=False)

            preprocessor = OmicsPreprocessor(
                scaling_method="identity", max_genes=-1, log_scaling=False, gene_list_source=str(gene_list_file)
            )

            transformed = preprocessor.fit_transform(sample_data)

            assert transformed.shape[1] == len(selected_genes)
            assert all(col in selected_genes for col in transformed.columns)

    def test_preprocessor_gene_list_source_list(self, sample_data):
        """Test gene selection from list."""
        selected_genes = ["gene_0", "gene_5", "gene_10"]

        preprocessor = OmicsPreprocessor(
            scaling_method="identity", max_genes=-1, log_scaling=False, gene_list_source=selected_genes
        )

        transformed = preprocessor.fit_transform(sample_data)

        assert transformed.shape[1] == len(selected_genes)

    def test_preprocessor_fit_then_transform(self, sample_data):
        """Test separate fit and transform calls."""
        preprocessor = OmicsPreprocessor(scaling_method="min_max", max_genes=-1, log_scaling=True)

        # Split data
        train_data = sample_data.iloc[:40]
        test_data = sample_data.iloc[40:]

        # Fit on train
        preprocessor.fit(train_data)

        # Transform both
        train_transformed = preprocessor.transform(train_data)
        test_transformed = preprocessor.transform(test_data)

        assert train_transformed.shape == train_data.shape
        assert test_transformed.shape == test_data.shape

    def test_preprocessor_columns_consistency(self, sample_data):
        """Test that columns are consistent between fit and transform."""
        preprocessor = OmicsPreprocessor(scaling_method="identity", max_genes=50, log_scaling=False)

        preprocessor.fit(sample_data)
        columns_after_fit = preprocessor.columns_to_keep.copy()

        transformed = preprocessor.transform(sample_data)

        assert preprocessor.columns_to_keep == columns_after_fit
        assert list(transformed.columns) == sorted(columns_after_fit)

    def test_preprocessor_rank_genes(self, sample_data):
        """Test gene ranking by variance."""
        preprocessor = OmicsPreprocessor(scaling_method="identity", max_genes=-1, log_scaling=False)

        gene_ranks = preprocessor.rank_genes(sample_data)

        # Check that all genes are ranked
        assert len(gene_ranks) == sample_data.shape[1]
        # Check that ranks are unique
        assert len(set(gene_ranks)) == len(gene_ranks)

    def test_all_scalers_work(self, sample_data):
        """Test that all scalers in SCALERS work."""
        for scaler_name in SCALERS:
            preprocessor = OmicsPreprocessor(scaling_method=scaler_name, max_genes=-1, log_scaling=False)

            transformed = preprocessor.fit_transform(sample_data)
            assert transformed.shape == sample_data.shape


class MockDataset:
    """Mock dataset helper for testing PreclinicalDataset methods."""

    def __init__(self):
        """Initialize mock dataset."""
        samples = [f"sample_{i}" for i in range(10)]
        perturbations = ["pert_A", "pert_B", "pert_C", "pert_D"]

        # Create dataframes
        self.df_labels = pd.DataFrame(
            np.random.randn(len(samples), len(perturbations)), index=samples, columns=perturbations
        )

        self.df_rnaseq = pd.DataFrame(
            np.random.randn(len(samples), 20), index=samples, columns=[f"gene_{i}_rnaseq" for i in range(20)]
        )

        self.df_sample_metadata = pd.DataFrame(
            {"tissue": ["Lung", "Lung", "Breast", "Breast", "Liver"] * 2, "domain": ["source"] * 10},
            index=samples,
        )

        self.df_fingerprints = pd.DataFrame(
            np.random.randint(0, 2, (len(perturbations), 8)),
            index=perturbations,
            columns=[f"pathway_{i}" for i in range(8)],
        )

        # Stacked version
        self.df_labels_stacked = self.df_labels.stack()
        self.df_labels_stacked.index.names = ["sample", "perturbation"]
        self.df_labels_stacked = pd.DataFrame(self.df_labels_stacked, columns=["label"])

        self.df_sample_metadata_stacked = pd.merge(
            self.df_sample_metadata,
            self.df_labels_stacked.reset_index(),
            left_index=True,
            right_on="sample",
        ).set_index(["sample", "perturbation"])

    def _sort_rows_and_columns(self):
        """Sort rows and columns for consistency."""
        self.df_labels = self.df_labels.sort_index(axis=0).sort_index(axis=1)
        if self.df_fingerprints is not None:
            self.df_fingerprints = self.df_fingerprints.sort_index(axis=0)

    def stack_dataframes(self):
        """Stack labels and align sample metadata."""
        self.df_labels_stacked = self.df_labels.stack()
        self.df_labels_stacked.index.names = ["sample", "perturbation"]
        self.df_labels_stacked = pd.DataFrame(self.df_labels_stacked, columns=["label"])

        self.df_sample_metadata_stacked = pd.merge(
            self.df_sample_metadata,
            self.df_labels_stacked.reset_index(),
            left_index=True,
            right_on="sample",
        ).set_index(["sample", "perturbation"])


class TestPreclinicalDataset:
    """Test PreclinicalDataset class methods."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock PreclinicalDataset for testing."""
        np.random.seed(42)
        return MockDataset()

    def test_keep_perturbations(self, mock_dataset):
        """Test keep_perturbations method."""
        from leap.data.preclinical_dataset import PreclinicalDataset

        # Copy the method to the mock
        mock_dataset.keep_perturbations = PreclinicalDataset.keep_perturbations.__get__(mock_dataset)

        # Keep only subset of perturbations
        to_keep = ["pert_A", "pert_C"]
        mock_dataset.keep_perturbations(to_keep)

        # Check that only selected perturbations remain
        assert list(mock_dataset.df_labels.columns) == to_keep
        assert list(mock_dataset.df_fingerprints.index) == to_keep

        # Check that stacked dataframe was updated
        assert mock_dataset.df_labels_stacked.shape[0] == len(to_keep) * len(mock_dataset.df_labels.index)

    def test_keep_perturbations_missing_fingerprints(self, mock_dataset):
        """Test keep_perturbations when some perturbations don't have fingerprints."""
        from leap.data.preclinical_dataset import PreclinicalDataset

        mock_dataset.keep_perturbations = PreclinicalDataset.keep_perturbations.__get__(mock_dataset)

        # Remove one perturbation from fingerprints
        mock_dataset.df_fingerprints = mock_dataset.df_fingerprints.drop("pert_D")

        # Keep all perturbations including the one without fingerprints
        to_keep = ["pert_A", "pert_B", "pert_C", "pert_D"]
        mock_dataset.keep_perturbations(to_keep)

        # Check that labels kept all
        assert list(mock_dataset.df_labels.columns) == to_keep

        # Check that fingerprints only has those available
        assert "pert_D" not in mock_dataset.df_fingerprints.index
        assert len(mock_dataset.df_fingerprints.index) == 3

    def test_keep_perturbations_no_fingerprints(self, mock_dataset):
        """Test keep_perturbations when fingerprints become empty."""
        from leap.data.preclinical_dataset import PreclinicalDataset

        mock_dataset.keep_perturbations = PreclinicalDataset.keep_perturbations.__get__(mock_dataset)

        # Keep perturbations that don't exist in fingerprints
        to_keep = ["pert_E", "pert_F"]
        mock_dataset.df_labels = mock_dataset.df_labels.copy()
        mock_dataset.df_labels["pert_E"] = 1.0
        mock_dataset.df_labels["pert_F"] = 2.0

        mock_dataset.keep_perturbations(to_keep)

        # Fingerprints should be None when empty
        assert mock_dataset.df_fingerprints is None

    def test_merge(self, mock_dataset):
        """Test merge method."""
        from leap.data.preclinical_dataset import PreclinicalDataset

        mock_dataset.merge = PreclinicalDataset.merge.__get__(mock_dataset)

        # Create another mock dataset to merge
        np.random.seed(100)
        other_dataset = MockDataset()

        # Modify the second dataset
        other_dataset.df_labels.index = [f"sample_new_{i}" for i in range(10)]
        other_dataset.df_rnaseq.index = [f"sample_new_{i}" for i in range(10)]
        other_dataset.df_sample_metadata.index = [f"sample_new_{i}" for i in range(10)]

        # Store original sizes
        original_n_samples = len(mock_dataset.df_labels)
        original_n_genes = len(mock_dataset.df_rnaseq.columns)

        # Merge
        mock_dataset.merge(other_dataset)

        # Check that datasets were concatenated
        assert len(mock_dataset.df_labels) == original_n_samples + 10
        assert len(mock_dataset.df_sample_metadata) == original_n_samples + 10

        # Check that stacked version was updated
        assert mock_dataset.df_labels_stacked.shape[0] > original_n_samples * 4

        # Check that rnaseq kept only common columns
        assert len(mock_dataset.df_rnaseq.columns) == original_n_genes

    def test_rename_for_code(self):
        """Test rename_for_code utility function."""
        from leap.data.preclinical_dataset import rename_for_code

        assert rename_for_code("Lung") == "lung"
        assert rename_for_code("Small Intestine") == "small_intestine"
        assert rename_for_code("Central-Nervous") == "central_nervous"
        assert rename_for_code("Head-and-Neck") == "head_and_neck"
        assert rename_for_code("T cell") == "t_cell"

    def test_filter_tissues_keep_specific(self, mock_dataset):
        """Test _filter_tissues method with specific tissues."""
        from leap.data.preclinical_dataset import PreclinicalDataset

        mock_dataset._filter_tissues = PreclinicalDataset._filter_tissues.__get__(mock_dataset)
        mock_dataset.tissues_to_keep = ["Lung", "Breast"]
        mock_dataset.tissues_to_exclude = None

        original_n_samples = len(mock_dataset.df_sample_metadata)

        # Filter
        mock_dataset._filter_tissues()

        # Should keep only Lung and Breast samples
        remaining_tissues = mock_dataset.df_sample_metadata["tissue"].unique()
        assert set(remaining_tissues) <= {"Lung", "Breast"}
        assert len(mock_dataset.df_sample_metadata) < original_n_samples

    def test_filter_tissues_exclude_specific(self, mock_dataset):
        """Test _filter_tissues method with excluded tissues."""
        from leap.data.preclinical_dataset import PreclinicalDataset

        mock_dataset._filter_tissues = PreclinicalDataset._filter_tissues.__get__(mock_dataset)
        mock_dataset.tissues_to_keep = "all"
        mock_dataset.tissues_to_exclude = ["Liver"]

        # Filter
        mock_dataset._filter_tissues()

        # Should not have Liver samples
        remaining_tissues = mock_dataset.df_sample_metadata["tissue"].unique()
        assert "Liver" not in remaining_tissues

    def test_filter_tissues_keep_all(self, mock_dataset):
        """Test _filter_tissues method keeping all tissues."""
        from leap.data.preclinical_dataset import PreclinicalDataset

        mock_dataset._filter_tissues = PreclinicalDataset._filter_tissues.__get__(mock_dataset)
        mock_dataset.tissues_to_keep = "all"
        mock_dataset.tissues_to_exclude = None

        original_n_samples = len(mock_dataset.df_sample_metadata)

        # Filter
        mock_dataset._filter_tissues()

        # Should keep all samples
        assert len(mock_dataset.df_sample_metadata) == original_n_samples
