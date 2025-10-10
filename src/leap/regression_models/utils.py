"""Elastic Net model for LEAP.

The ElasticNet model is already implemented in skglm, so here we just define a utils for the alpha grid search.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model._coordinate_descent import _alpha_grid
from torch import nn


class SpearmanLoss(nn.Module):
    """Differentiable Surrogate Spearman correlation loss module.

    A PyTorch module wrapper for the differentiable Spearman correlation loss.
    This computes 1 - correlation to convert it into a loss (minimization problem).

    Note
    ----
    Despite the name "Spearman" being used in the configuration, this actually computes Pearson correlation (not
    Spearman rank correlation). For true Spearman correlation, the inputs would need to be ranked first.
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the Spearman loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True target values.

        Returns
        -------
        torch.Tensor
            Differentiable Surrogate Spearman loss (1 - correlation).
        """
        # Compute the covariance
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        cov = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))

        # Compute the standard deviations
        y_true_std = torch.std(y_true)
        y_pred_std = torch.std(y_pred)

        # Compute the correlation
        spearman_corr = cov / (y_true_std * y_pred_std + 1e-8)  # Add epsilon to avoid division by zero

        # Return 1 - spearman_corr to convert it into a loss (minimization problem)
        return 1 - spearman_corr


class AlphaGridElasticNet:
    """Class for the grid of alpha parameters.

    Parameters
    ----------
    alpha_min_ratio : float
        The ratio to define the minimum alpha parameter.
    n_alphas : int
        The number of alpha parameters.
    """

    def __init__(self, alpha_min_ratio: float = 1e-3, n_alphas: int = 10):
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alphas = n_alphas

    def get_alpha_grid(self, X: pd.DataFrame, y: pd.DataFrame, l1_ratio: float) -> list:
        """Define the grid of alpha parameters.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.DataFrame
            The output data.
        l1_ratio: float
            The l1_ratio of the Elastic Net model.

        Returns
        -------
        list
            The list of alpha parameters.
        """
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_np = y.to_numpy().ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else y
        alpha_max = _alpha_grid(X_np, y_np, l1_ratio=l1_ratio, n_alphas=1)[0]
        alpha_grid = list(np.logspace(np.log10(alpha_max * self.alpha_min_ratio), np.log10(alpha_max), self.n_alphas))
        return alpha_grid
