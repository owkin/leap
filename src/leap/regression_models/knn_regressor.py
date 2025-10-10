"""KNN Regression Model."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor


class KnnRegressor(BaseEstimator):
    """KNN Regression Model.

    This KNN model is using the KNeighborsRegressor from scikit-learn in a way that allows having nan values in the
    target variable. The model is fit for each target on the X data that correspond to non-nan values in the target
    variable.

    Parameters
    ----------
    n_sample_neighbors : int
        Number of neighbors to use.
    weights : str
        Weight function used in prediction.
    n_jobs : int
        Number of jobs to run in parallel. Default is 1.
    """

    def __init__(self, n_sample_neighbors: int, weights: str, n_jobs: int = 1):
        super().__init__()
        self.n_sample_neighbors = n_sample_neighbors
        self.weights = weights
        self.n_jobs = n_jobs
        self.X_train: pd.DataFrame
        self.y_train: pd.DataFrame

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs: Any) -> None:
        """Fit method for the KnnRegressor.

        KNeighborsRegressor is a single label regression model but instantiating and storing a unique model per
        perturbation (times N splits and M repeats) is way too expensive in terms of memory and time. Therefore we
        suggest to only store the training data at training time and to fit the knn at every inference call.
        This is possible as KNN is a non-parametric model and inference is fast.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_pred: pd.DataFrame) -> np.ndarray:
        """Predict method.

        The fact that the `fit` method of KNeighborsRegressor is called here makes it more memory efficient than storing
        a unique model per perturbation.
        """
        pred_values = []
        for col_target in self.y_train.columns:
            y_train_col = self.y_train[col_target].dropna()
            X_train_col = self.X_train.loc[y_train_col.index]

            # hack to avoid having n_sample_neighbors higher than the number of samples
            # Note: the spearman will then be 0 by our definition of the metric when
            # the prediction is constant.
            n_sample_neighbors = min(self.n_sample_neighbors, len(y_train_col))

            knn = KNeighborsRegressor(n_neighbors=n_sample_neighbors, weights=self.weights, n_jobs=self.n_jobs)
            knn.fit(X_train_col, y_train_col)
            pred_values.append(knn.predict(X_pred).tolist())
        return np.array(pred_values).T
