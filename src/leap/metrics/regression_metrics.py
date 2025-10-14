"""Regression metrics."""

import warnings
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


REGRESSION_METRICS = ("spearman", "pearson", "r2", "mse", "mae")
RegressionMetricType: TypeAlias = Literal["spearman", "pearson", "r2", "mse", "mae"]


def performance_metric_wrapper(
    y_true: pd.Series, y_pred: pd.Series, metric: RegressionMetricType = "spearman", per_perturbation: bool = False
) -> float:
    """Calculate a performance metric.

    This function accommodates missing values in the true labels: these are excluded before calculating the performances

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : pd.Series
        Predicted labels.
    metric : RegressionMetricType, optional
        Metric to compute, by default "spearman". Possible values are: "spearman", "pearson", "r2", "mse" and "mae".
    per_perturbation : bool, optional
        Whether to compute the performance metric per perturbation, by default False.

    Returns
    -------
    float
        Performance metric.
    """
    # Exclude missing true labels
    y_true = y_true.dropna()
    y_pred = y_pred[y_true.index]

    # Compute the performance metric
    if per_perturbation:
        # Calculate the average per-perturbation metric
        return (
            pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            .groupby("perturbation")
            .apply(lambda x: performance_metric(x["y_true"].to_numpy(), x["y_pred"].to_numpy(), metric=metric))
        ).mean()

    # Calculate the overall metric
    return performance_metric(y_true.to_numpy(), y_pred.to_numpy(), metric)


def performance_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: RegressionMetricType = "spearman") -> float:
    """Calculate a performance metric for a continuous label. Correlation metrics for constant outputs are set to 0.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    metric : RegressionMetricType, optional
        Metric to compute, by default "spearman". Possible values are: "spearman", "pearson", "r2", "mse" and "mae".

    Returns
    -------
    float
        Performance metric.

    Raises
    ------
    ValueError
        If the metric is not implemented.
    """
    if metric == "r2":
        return r2_score(y_true, y_pred)
    if metric == "mse":
        return mse(y_true, y_pred)
    if metric == "mae":
        return mae(y_true, y_pred)
    # For correlation metrics, we set the metric to 0 if the outputs are constant
    if metric == "spearman":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nan_to_num(spearmanr(y_true, y_pred, nan_policy="omit")[0])
    if metric == "pearson":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nan_to_num(pearsonr(y_true, y_pred)[0])

    raise ValueError(f"Unsupported metric: {metric}. Must be one of {REGRESSION_METRICS}")
