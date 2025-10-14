"""Masked Autoencoder implementation for self-supervised pre-training.

This module implements the Masked Autoencoder as described in:
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain" (2020)
"""

from collections import defaultdict
from typing import Any, Literal

import torch

from .auto_encoder import AutoEncoder


class MaskedAutoencoder(AutoEncoder):
    """Masked Autoencoder for self-supervised pre-training.

    Inherits from AutoEncoder and adds:
    1. Pretraining with masking/corruption
    2. Option to train with reconstruction and mask prediction loss

    Parameters
    ----------
    beta : float
        Noise level (used when corruption_method is set to "noise").
    corruption_proba : float
        Masking/corruption probability.
    corruption_method : Literal["classic", "vime", "noise", "full_noise"]
        Type of corruption to use:
            - "classic": Regular masking (zeros)
            - "vime": Permutation-based corruption
            - "noise": Additive Gaussian noise
            - "full_noise": Full Gaussian noise replacement
    data_augmentation : bool
        Whether to add Gaussian noise to the input data as augmentation.
    da_noise_std : float
        Standard deviation of the Gaussian noise added to the input data when data_augmentation is True.


    Attributes
    ----------
    losses : dict[str, list[float]]
        Dictionary storing different types of losses during training.
    metrics : dict[str, list[float]]
        Dictionary storing different metrics during training.

    Raises
    ------
    ValueError
        If corruption_method is not one of the allowed values.
        If data_augmentation is True but corruption_method is not compatible.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from leap.representation_models import MaskedAutoencoder
    >>>
    >>> X = pd.DataFrame(np.random.randn(100, 50))
    >>> mae = MaskedAutoencoder(repr_dim=10, corruption_proba=0.3, corruption_method="vime")
    >>> X_transformed = mae.fit(X)
    >>> X_transformed = mae.transform(X)
    """

    def __init__(
        self,
        beta: float = 0.1,
        corruption_proba: float = 0.3,
        corruption_method: Literal["classic", "vime", "noise", "full_noise"] = "classic",
        data_augmentation: bool = False,
        da_noise_std: float = 0.01,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Mask configuration
        self.corruption_proba = corruption_proba
        self.beta = beta
        self.losses: dict[str, list[float]] = defaultdict(list)
        self.metrics: dict[str, list[float]] = defaultdict(list)

        # Validate corruption method
        valid_methods = ["classic", "vime", "noise", "full_noise"]
        if corruption_method not in valid_methods:
            raise ValueError(f"corruption_method must be one of {valid_methods}, got '{corruption_method}'")
        self.corruption_method = corruption_method

        # Data augmentation configuration
        self.data_augmentation = data_augmentation
        self.da_noise_std = da_noise_std

    def mask_generator(self, x: torch.Tensor) -> torch.Tensor:
        """Generate random mask vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Binary mask tensor with corruption_proba chance of 1.
        """
        tensor_p = torch.ones_like(x) * self.corruption_proba
        mask = torch.bernoulli(tensor_p)
        return mask

    def pretext_generator(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Generate corrupted samples according to the corruption method.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask indicating which elements to corrupt.
        x : torch.Tensor
            Original input tensor.

        Returns
        -------
        torch.Tensor
            Corrupted input tensor.
        """
        n, dim = x.shape

        if self.corruption_method == "vime":
            # Permutation-based corruption
            perm = torch.randperm(x.size(0), device=x.device)
            x_bar = x[perm]
        elif self.corruption_method == "noise":
            # Additive Gaussian noise
            x_noise = torch.randn_like(x)
            x_bar = x + self.beta * x_noise
        elif self.corruption_method == "full_noise":
            # Full Gaussian noise replacement
            x_bar = torch.randn_like(x)
        else:  # "classic"
            # Zero masking
            x_bar = torch.zeros([n, dim], device=self.device)

        # Apply mask: keep original where mask=0, corrupt where mask=1
        x_tilde = x * (1 - mask) + x_bar * mask
        return x_tilde

    def forward(self, x: torch.Tensor, eval_mode: bool = False, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the masked autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        eval_mode : bool
            If True, skip masking (standard autoencoder reconstruction).

        Returns
        -------
        torch.Tensor
            Reconstructed/decoded tensor.

        Raises
        ------
        ValueError
            If data_augmentation is True but corruption_method is not compatible.
        """
        if eval_mode:
            # Standard reconstruction without masking
            return self.decoder(self.encoder(x))

        # Apply data augmentation if enabled
        if self.data_augmentation:
            # Validate compatible corruption methods
            if self.corruption_method not in ["classic", "vime", "full_noise"]:
                raise ValueError(
                    "data_augmentation can only be used with corruption_method 'classic', 'vime', or 'full_noise'"
                )
            x = x + self.da_noise_std * torch.randn_like(x)

        # Generate mask and corrupted input
        mask = self.mask_generator(x)
        x_tilde = self.pretext_generator(mask, x)

        # Encode and decode
        return self.decoder(self.encoder(x_tilde))
