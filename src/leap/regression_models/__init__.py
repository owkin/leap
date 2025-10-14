"""Prediction models for LEAP."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class RegressionModel(ABC):
    """Abstract base class defining the interface for regression models in LEAP.

    All regression models must inherit from this class and implement fit(), predict(),
    set_params(), and get_params() methods with these signatures.
    """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def set_params(self, **kwargs: Any) -> "RegressionModel":
        """Set parameters for this estimator.

        Returns
        -------
        RegressionModel
            The instance itself.
        """
        ...

    @abstractmethod
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


# Import classes that depend on RegressionModel AFTER defining it
from .elastic_net import ElasticNet  # noqa: E402
from .knn_regressor import KnnRegressor  # noqa: E402
from .lgbm_regressor import LGBMRegressor  # noqa: E402
from .mlp_regressor import TorchMLPRegressor  # noqa: E402
from .utils import AlphaGridElasticNet  # noqa: E402


__all__ = [
    "AlphaGridElasticNet",
    "ElasticNet",
    "KnnRegressor",
    "LGBMRegressor",
    "RegressionModel",
    "TorchMLPRegressor",
]
