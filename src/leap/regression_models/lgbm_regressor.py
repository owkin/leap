"""LGBMRegressor wrapper for LEAP that conforms to the RegressionModel interface."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor as LightGBMRegressor

# RegressionModel is defined in __init__.py before this module is imported
from leap.regression_models import RegressionModel


class LGBMRegressor(RegressionModel):
    """Wrapper for LightGBM's LGBMRegressor that accepts pandas DataFrames.

    This wrapper ensures compatibility with LEAP's RegressionModel interface by:
    - Accepting pandas DataFrames for fit() and predict()
    - Converting internally to numpy arrays as required by LightGBM
    - Providing a consistent interface with other LEAP regression models
    """

    def __init__(self, **kwargs: Any):
        self._model = LightGBMRegressor(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, **kwargs: Any) -> None:
        """Fit the LGBMRegressor model.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series | pd.DataFrame
            Training targets.
        """
        # Keep X as DataFrame to preserve feature names in LightGBM
        # Only flatten y to avoid shape issues
        y_array = y.values.ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else y

        # Fit the underlying model
        self._model.fit(X, y_array)

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
        # Keep X as DataFrame to match feature names from fit
        return self._model.predict(X)

    def set_params(self, **params: Any) -> LGBMRegressor:
        """Set the parameters of this estimator.

        Returns
        -------
        LGBMRegressor
            The instance itself.
        """
        self._model.set_params(**params)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.
        """
        return self._model.get_params(deep=deep)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            The attribute value from the underlying model.

        Raises
        ------
        AttributeError
            If the attribute is not found.
        """
        # Avoid infinite recursion during copy/pickle by not delegating special methods
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if _model exists to avoid recursion during initialization/copying
        if "_model" not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return getattr(self._model, name)
