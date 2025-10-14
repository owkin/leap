"""Utility functions for representation models."""

import numpy as np
import torch


class OmicsDataset(torch.utils.data.Dataset):
    """OmicsDataset Dataset Class for PyTorch models.

    Parameters
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray | None
        Labels, by default None.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Get a single sample from the dataset."""
        features = torch.from_numpy(self.X[item].astype(np.float32))
        if self.y is not None:
            label = torch.Tensor([self.y[item]]).float()
            return features, label
        return features


def _initialize_early_stopping(eval_loss: list[float], early_stopping_best: float) -> float:
    """Initialize the current best loss at the first epoch.

    Only useful for the AE, as other models' metrics can be initialized to their worst value (e.g., 0).

    Parameters
    ----------
    eval_loss : list[float]
        The list of loss at every epoch.
    early_stopping_best : float
        The best validation performance so far.

    Returns
    -------
    float
        Either the initialized early_stopping_best or the current one.
    """
    if len(eval_loss) == 1:  # First epoch
        early_stopping_best = eval_loss[0]
    return early_stopping_best


def _update_early_stopping(
    eval_list: list[float],
    early_stopping_best: float,
    early_stopping_delta: float,
    early_stopping_patience_count: int,
    use_metric: bool = False,
) -> tuple[float, int]:
    """Update the best loss/metric value epoch by epoch, along with the patience count.

    If we use a metric (higher is better), we want it to increase. If we use the loss (lower is better), we want it to
    decrease.

    Parameters
    ----------
    eval_list : list[float]
        Either the metric or the loss at each epoch.
    early_stopping_best : float
        The current best performance on the eval set.
    early_stopping_delta : float
        The threshold for which we consider the model hasn't improved enough.
    early_stopping_patience_count : int
        The current number of epochs the model hasn't improved enough.
    use_metric : bool
        Whether a metric is used (True) or the loss (False).

    Returns
    -------
    tuple[float, int]
        The updated early_stopping_best and early_stopping_patience_count.
    """
    if use_metric and (eval_list[-1] > early_stopping_best + early_stopping_delta):
        early_stopping_best = eval_list[-1]
        early_stopping_patience_count = 0
    elif not use_metric and (eval_list[-1] < early_stopping_best - early_stopping_delta):
        early_stopping_best = eval_list[-1]
        early_stopping_patience_count = 0
    else:
        early_stopping_patience_count += 1

    return (early_stopping_best, early_stopping_patience_count)
