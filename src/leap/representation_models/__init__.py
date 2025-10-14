"""Representation models for LEAP."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class RepresentationModelBase(ABC):
    """Abstract base class for representation models in LEAP.

    All representation models must implement fit(), transform() methods and define
    a repr_dim attribute in their __init__ method that specifies the dimensionality
    of the learned representation.

    This base class works with sklearn-based models (PCA) and custom PyTorch
    implementations (AutoEncoder, MaskedAutoencoder) alike.

    Attributes
    ----------
    repr_dim : int
        The dimensionality of the learned representation. All subclasses must set
        this attribute in their __init__ method.

    Notes
    -----
    While `repr_dim` cannot be enforced at the abstract class level due to being
    set in __init__, all implementations should include this attribute. The type
    checker will help ensure this requirement is met.
    """

    # Type hint for the attribute that subclasses must set
    repr_dim: int

    @abstractmethod
    def fit(self, X: pd.DataFrame, **kwargs: Any) -> "RepresentationModelBase":
        """Fit the representation model to the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.

        Returns
        -------
        RepresentationModelBase
            The fitted model instance.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement fit().
        """
        raise NotImplementedError("Subclasses must implement fit()")

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using the fitted representation model.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        np.ndarray
            Transformed data in the learned representation space.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement transform().
        """
        raise NotImplementedError("Subclasses must implement transform()")


# Import classes that depend on RepresentationModelBase AFTER defining it
from .auto_encoder import AutoEncoder  # noqa: E402
from .masked_auto_encoder import MaskedAutoencoder  # noqa: E402
from .pca import PCA  # noqa: E402


__all__ = ["PCA", "AutoEncoder", "MaskedAutoencoder", "RepresentationModelBase"]
