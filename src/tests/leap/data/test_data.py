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
