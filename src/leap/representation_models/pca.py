"""Dimension reduction method based on principal component analysis."""

from typing import Any

import numpy as np
import pandas as pd
import sklearn.decomposition

from . import RepresentationModelBase


class PCA(RepresentationModelBase, sklearn.decomposition.PCA):
    """Principal Component Analysis for dimensionality reduction.

    This class extends sklearn's PCA with a consistent interface for LEAP, including support for pandas DataFrames.

    Parameters
    ----------
    repr_dim : int
        Number of dimensions for the dimension reduction (number of components).
    random_state : int
        Seed for the random number generator.

    Attributes
    ----------
    repr_dim : int
        Number of principal components.
    components_ : np.ndarray
        Principal axes in feature space.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from leap.representation_models import PCA
    >>>
    >>> X = pd.DataFrame(np.random.randn(100, 50))
    >>> pca = PCA(repr_dim=10)
    >>> pca.fit(X)
    >>> X_transformed = pca.transform(X)
    >>> print(X_transformed.shape)
    (100, 10)
    """

    def __init__(self, repr_dim: int, random_state: int = 42):
        super().__init__(n_components=repr_dim, random_state=random_state)
        self.repr_dim = repr_dim

    def fit(self, X: pd.DataFrame, **kwargs: Any) -> "PCA":
        """Fit the PCA model with X.

        Parameters
        ----------
        X : pd.DataFrame
            Training matrix of shape (n_samples, n_features).

        Returns
        -------
        PCA
            The fitted instance.
        """
        # Convert DataFrame to numpy array for sklearn
        # super() calls sklearn.decomposition.PCA.fit() due to MRO
        super().fit(X)  # type: ignore[safe-super]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed values, shape (n_samples, n_components).
        """
        # super() calls sklearn.decomposition.PCA.transform() due to MRO
        return super().transform(X)  # type: ignore[safe-super]
