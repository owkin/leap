"""Prediction models for LEAP."""

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skglm import ElasticNet

from .knn_regressor import KnnRegressor
from .mlp_regressor import TorchMLPRegressor
from .utils import AlphaGridElasticNet


@runtime_checkable
class RegressionModel(Protocol):
    """Protocol defining the interface for regression models in LEAP.

    All regression models must implement fit(), predict() and set_params() methods with these signatures. This protocol
    works with external libraries (ElasticNet, LGBMRegressor) and custom implementations alike.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, **kwargs: Any) -> None:
        """Fit the regression model.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series | pd.DataFrame
            Training targets.
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        ...

    def set_params(self, **kwargs: Any) -> "RegressionModel":
        """Set parameters for this estimator.

        Returns
        -------
        RegressionModel
            The instance itself.
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.
        """
        ...


__all__ = [
    "AlphaGridElasticNet",
    "ElasticNet",
    "KnnRegressor",
    "LGBMRegressor",
    "RegressionModel",
    "TorchMLPRegressor",
]
