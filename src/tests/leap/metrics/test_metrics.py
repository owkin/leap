"""Tests for the metrics module."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

from leap.metrics.regression_metrics import (
    REGRESSION_METRICS,
    performance_metric,
    performance_metric_wrapper,
)


class TestPerformanceMetric:
    """Test performance_metric function."""

    def test_spearman_correlation(self):
        """Test Spearman correlation metric."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

        result = performance_metric(y_true, y_pred, metric="spearman")
        expected, _ = spearmanr(y_true, y_pred)

        assert abs(result - expected) < 1e-6

    def test_pearson_correlation(self):
        """Test Pearson correlation metric."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

        result = performance_metric(y_true, y_pred, metric="pearson")
        expected, _ = pearsonr(y_true, y_pred)

        assert abs(result - expected) < 1e-6

    def test_r2_score(self):
        """Test R² score metric."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

        result = performance_metric(y_true, y_pred, metric="r2")

        # R² should be high for good predictions
        assert result > 0.9

    def test_mse(self):
        """Test Mean Squared Error metric."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        result = performance_metric(y_true, y_pred, metric="mse")

        # MSE should be 0 for perfect predictions
        assert result == 0.0

    def test_mae(self):
        """Test Mean Absolute Error metric."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        result = performance_metric(y_true, y_pred, metric="mae")

        # MAE should be 0 for perfect predictions
        assert result == 0.0

    def test_constant_predictions_spearman(self):
        """Test that constant predictions return 0 for Spearman."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 2, 2, 2, 2])  # Constant predictions

        result = performance_metric(y_true, y_pred, metric="spearman")

        # Spearman correlation for constant predictions should be 0
        assert result == 0.0

    def test_constant_predictions_pearson(self):
        """Test that constant predictions return 0 for Pearson."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 2, 2, 2, 2])  # Constant predictions

        result = performance_metric(y_true, y_pred, metric="pearson")

        # Pearson correlation for constant predictions should be 0
        assert result == 0.0

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

        with pytest.raises(ValueError, match="Unsupported metric"):
            performance_metric(y_true, y_pred, metric="invalid_metric")

    def test_all_metrics_are_valid(self):
        """Test that all metrics in REGRESSION_METRICS work."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

        for metric in REGRESSION_METRICS:
            result = performance_metric(y_true, y_pred, metric=metric)
            assert isinstance(result, (float, np.floating))


class TestPerformanceMetricWrapper:
    """Test performance_metric_wrapper function."""

    def test_wrapper_with_series(self):
        """Test wrapper with pandas Series."""
        y_true = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
        y_pred = pd.Series([1.1, 2.2, 2.9, 4.1, 5.2], index=["a", "b", "c", "d", "e"])

        result = performance_metric_wrapper(y_true, y_pred, metric="spearman")

        assert isinstance(result, (float, np.floating))
        assert result > 0.9

    def test_wrapper_handles_missing_values(self):
        """Test that wrapper correctly handles missing values."""
        y_true = pd.Series([1, 2, np.nan, 4, 5], index=["a", "b", "c", "d", "e"])
        y_pred = pd.Series([1.1, 2.2, 2.9, 4.1, 5.2], index=["a", "b", "c", "d", "e"])

        result = performance_metric_wrapper(y_true, y_pred, metric="spearman")

        # Should compute metric only on non-missing values
        assert isinstance(result, (float, np.floating))

    def test_wrapper_with_multiindex(self):
        """Test wrapper with MultiIndex (sample, perturbation)."""
        index = pd.MultiIndex.from_tuples(
            [("sample1", "geneA"), ("sample1", "geneB"), ("sample2", "geneA"), ("sample2", "geneB")],
            names=["sample", "perturbation"],
        )
        y_true = pd.Series([1, 2, 3, 4], index=index)
        y_pred = pd.Series([1.1, 2.1, 3.1, 4.1], index=index)

        # Test overall metric
        result = performance_metric_wrapper(y_true, y_pred, metric="spearman", per_perturbation=False)
        assert isinstance(result, (float, np.floating))

    def test_wrapper_per_perturbation(self):
        """Test per-perturbation metric calculation."""
        index = pd.MultiIndex.from_tuples(
            [
                ("sample1", "geneA"),
                ("sample2", "geneA"),
                ("sample3", "geneA"),
                ("sample1", "geneB"),
                ("sample2", "geneB"),
                ("sample3", "geneB"),
            ],
            names=["sample", "perturbation"],
        )
        y_true = pd.Series([1, 2, 3, 4, 5, 6], index=index)
        y_pred = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1, 6.1], index=index)

        result = performance_metric_wrapper(y_true, y_pred, metric="spearman", per_perturbation=True)

        # Should return average across perturbations
        assert isinstance(result, (float, np.floating))

    def test_wrapper_different_metrics(self):
        """Test wrapper with different metrics."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1.1, 2.2, 2.9, 4.1, 5.2])

        for metric in REGRESSION_METRICS:
            result = performance_metric_wrapper(y_true, y_pred, metric=metric)
            assert isinstance(result, (float, np.floating))

    def test_wrapper_perfect_prediction(self):
        """Test wrapper with perfect predictions."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = pd.Series([1, 2, 3, 4, 5])

        spearman_result = performance_metric_wrapper(y_true, y_pred, metric="spearman")
        pearson_result = performance_metric_wrapper(y_true, y_pred, metric="pearson")
        mse_result = performance_metric_wrapper(y_true, y_pred, metric="mse")

        assert abs(spearman_result - 1.0) < 1e-10
        assert abs(pearson_result - 1.0) < 1e-10
        assert abs(mse_result - 0.0) < 1e-10
