"""AutoEncoder implementation using PyTorch."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from leap.utils.seed import seed_everything

from . import RepresentationModelBase
from .utils import OmicsDataset, _initialize_early_stopping, _update_early_stopping


class AutoEncoder(RepresentationModelBase, torch.nn.Module):
    """Representation model using Autoencoder architecture.

    Parameters
    ----------
    repr_dim : int
        Size of the representation dimension (bottleneck).
    hidden_n_layers : int
        Number of hidden layers.
    hidden_n_units_first : int
        Number of units of the first hidden layer.
    hidden_decrease_rate : float
        Decrease rate of the number of units per layer.
    dropout : float | None
        Dropout probability for hidden layers. If None, no dropout is applied.
    activation : torch.nn.Module | None
        Activation function for the hidden layers.
    bias : bool
        If False, the layers will not learn an additive bias.
    num_epochs : int
        Number of epochs when early stopping is not used.
    batch_size : int
        Size of minibatches for training.
    learning_rate : float
        Learning rate for the optimizer.
    early_stopping_use : bool
        Whether to use early stopping with a validation set.
    max_num_epochs : int
        Maximum number of epochs when using early stopping.
    early_stopping_split : float
        Train/val split proportion for early stopping.
    early_stopping_patience : int
        Number of epochs without improvement before stopping.
    early_stopping_delta : float
        Minimum improvement required to reset patience counter.
    retrain : bool
        Whether to retrain on full data after early stopping finds optimal epochs.
    device : str
        Device to run training on ('cpu' or 'cuda').
    random_state : int
        Random seed for reproducibility.
    criterion : torch.nn.Module
        Loss function for the autoencoder reconstruction task.
    optimizer : Callable
        Optimizer class to use for training.

    Attributes
    ----------
    encoder : torch.nn.Module
        The encoder network.
    decoder : torch.nn.Module
        The decoder network.
    train_loss : list[float]
        Training loss history.
    eval_loss : list[float]
        Validation loss history (if early stopping is used).

    Raises
    ------
    ValueError
        If early_stopping_patience >= max_num_epochs.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from leap.representation_models import AutoEncoder
    >>>
    >>> X = pd.DataFrame(np.random.randn(100, 50))
    >>> ae = AutoEncoder(repr_dim=10, num_epochs=50)
    >>> X_transformed = ae.fit(X)
    >>> X_transformed = ae.transform(X)
    >>> print(X_transformed.shape)
    (100, 10)
    """

    def __init__(
        self,
        repr_dim: int,
        hidden_n_layers: int = 2,
        hidden_n_units_first: int = 512,
        hidden_decrease_rate: float = 0.5,
        dropout: float | None = None,
        activation: torch.nn.Module | None = torch.nn.ReLU(),
        bias: bool = True,
        num_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1.0e-4,
        early_stopping_use: bool = True,
        max_num_epochs: int = 1000,
        early_stopping_split: float = 0.2,
        early_stopping_patience: int = 50,
        early_stopping_delta: float = 0.001,
        retrain: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        random_state: int = 42,
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        optimizer: Callable = torch.optim.Adam,
    ):
        if early_stopping_use and early_stopping_patience >= max_num_epochs:
            raise ValueError("early_stopping_patience must be less than max_num_epochs")

        super().__init__()

        self.random_state = random_state
        seed_everything(self.random_state)
        self.repr_dim = repr_dim
        self.hidden_n_layers = hidden_n_layers
        self.hidden_n_units_first = hidden_n_units_first
        self.hidden_decrease_rate = hidden_decrease_rate
        self.hidden = self._convert_hidden_config()
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.num_epochs = max_num_epochs if early_stopping_use else num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_use = early_stopping_use
        self.early_stopping_split = early_stopping_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.retrain = retrain

        # Attributes initialized during fit
        self.in_features = 0
        self.feature_names_in_: list[str] | None = None
        self.train_loss: list[float]
        self.eval_loss: list[float]
        self.early_stopping_epoch: int
        self.encoder_early_stopping: torch.nn.Module | None = None
        self.decoder_early_stopping: torch.nn.Module | None = None
        self.encoder: torch.nn.Module
        self.decoder: torch.nn.Module
        self.early_stopping_best: float

    def _validate_feature_names(self, X: pd.DataFrame) -> None:
        """Validate that feature names match those seen during fit.

        Parameters
        ----------
        X : pd.DataFrame
            Data to validate.

        Raises
        ------
        ValueError
            If feature names don't match those seen during fit.
        """
        if self.feature_names_in_ is None:
            raise ValueError("This AutoEncoder instance is not fitted yet. Call 'fit' before using this method.")

        X_feature_names = X.columns.tolist()
        if X_feature_names != self.feature_names_in_:
            raise ValueError(
                f"The feature names should match those that were passed during fit.\n"
                f"Feature names seen during fit: {self.feature_names_in_}\n"
                f"Feature names seen now: {X_feature_names}"
            )

    def _convert_hidden_config(self) -> list[int]:
        """Convert hidden layer config to list of layer sizes.

        Converts from the 3 hidden layer configs (n_layers, n_units_first, and decrease_rate) to the traditional hidden
        list containing the number of nodes in each hidden layer.

        Returns
        -------
        list[int]
            List of hidden layer sizes.
        """
        hidden: list[int] = []

        if self.hidden_n_layers == 0:
            return hidden

        hidden.append(self.hidden_n_units_first)
        for i in range(1, self.hidden_n_layers):
            hidden.append(int(hidden[i - 1] * self.hidden_decrease_rate))

        return hidden

    def _init_models(self) -> None:
        """Initialize the encoder/decoder neural networks.

        This method is separate from __init__ because it depends on the number of features in the training data X (the
        shape of the first encoder layer and last decoder layer). This is called at each new fit() call.
        """
        encoder_output_sizes = [*self.hidden, self.repr_dim]
        decoder_output_sizes = self.hidden[::-1] if len(self.hidden) > 0 else []
        decoder_output_sizes.append(self.in_features)

        in_features_layer = self.in_features

        encoder_layers = []
        for i, size_of_layer_i in enumerate(encoder_output_sizes):
            layer_args: list[torch.nn.Module] = [
                torch.nn.Linear(in_features=in_features_layer, out_features=size_of_layer_i, bias=self.bias)
            ]
            in_features_layer = size_of_layer_i

            # Don't add activation/dropout on the last layer (bottleneck)
            if (self.activation is not None) and (i + 1 != len(encoder_output_sizes)):
                layer_args.append(self.activation)

            if (self.dropout is not None) and (i + 1 != len(encoder_output_sizes)):
                layer_args.append(torch.nn.Dropout(self.dropout))

            encoder_layers.append(torch.nn.Sequential(*layer_args))

        decoder_layers = []
        for i, size_of_layer_i in enumerate(decoder_output_sizes):
            layer_args = [torch.nn.Linear(in_features=in_features_layer, out_features=size_of_layer_i, bias=self.bias)]
            in_features_layer = size_of_layer_i

            # Don't add activation/dropout on the last layer (output)
            if (self.activation is not None) and (i + 1 != len(decoder_output_sizes)):
                layer_args.append(self.activation)

            if (self.dropout is not None) and (i + 1 != len(decoder_output_sizes)):
                layer_args.append(torch.nn.Dropout(self.dropout))

            decoder_layers.append(torch.nn.Sequential(*layer_args))

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute the encoding and decoding of the input x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed output.
        """
        return self.decoder(self.encoder(x))

    def fit(
        self,
        X: pd.DataFrame,
        metrics_suffix: str | None = None,
        **kwargs: Any,
    ) -> "AutoEncoder":
        """Fit the autoencoder model to the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        metrics_suffix : str | None
            Suffix for logging metrics (e.g., repeat/split information).

        Returns
        -------
        AutoEncoder
            The fitted autoencoder instance.
        """
        # Avoid printing "None" in logs
        metrics_suffix = f" {metrics_suffix}" if metrics_suffix else ""

        # Reset loss tracking
        self.train_loss, self.eval_loss = [], []

        # Store feature names for validation during transform
        if self.feature_names_in_ is None:
            self.feature_names_in_ = X.columns.tolist()

        if self.early_stopping_use:
            X_full = X.copy()
            X, X_val = train_test_split(X, test_size=self.early_stopping_split, random_state=self.random_state)

        dataset = OmicsDataset(X.values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        sample = next(iter(dataloader))
        self.in_features = sample.shape[1]
        self._init_models()

        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        early_stopping_best, early_stopping_patience_count = 0.0, 0
        disable_bar = True
        pbar = tqdm(range(self.num_epochs), total=self.num_epochs, disable=disable_bar)

        for epoch in pbar:
            # Set to train mode (because evaluate() calls eval())
            self.train()
            train_losses = []

            for data_batch in dataloader:
                data_batch = data_batch.to(self.device)

                data_batch_reconstructed = self.forward(data_batch)
                loss = self.criterion(data_batch_reconstructed, data_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = loss.detach().cpu().numpy()
                train_losses.append(train_loss)

            self.train_loss.append(np.mean(train_losses))
            pbar.set_description(f"train loss: {np.round(self.train_loss[-1], 4)!s}")

            if self.early_stopping_use:
                with torch.no_grad():
                    # Update the eval_loss at each epoch
                    self.evaluate(X_val)

                early_stopping_best = _initialize_early_stopping(self.eval_loss, early_stopping_best)

                (early_stopping_best, early_stopping_patience_count) = _update_early_stopping(
                    self.eval_loss, early_stopping_best, self.early_stopping_delta, early_stopping_patience_count
                )

                if early_stopping_patience_count > self.early_stopping_patience:
                    logger.info(f"AE training finished by early stopping at epoch {epoch + 1}")
                    self.early_stopping_epoch = epoch + 1
                    break

        else:  # No break occurred - finished all epochs
            logger.info(f"AE training finished with the max epoch number: {epoch + 1}")
            self.early_stopping_epoch = self.num_epochs

        if self.early_stopping_use:
            self.early_stopping_use = False
            self.num_epochs = self.early_stopping_epoch - self.early_stopping_patience
            self.early_stopping_best = early_stopping_best
            # Retrain on full data if requested
            if self.retrain:
                self.fit(X_full, metrics_suffix=metrics_suffix + " retrain")

        return self

    def evaluate(self, X: pd.DataFrame) -> "AutoEncoder":
        """Evaluate the model performance on validation data.

        Updates self.eval_loss by computing reconstruction loss on X.

        Parameters
        ----------
        X : pd.DataFrame
            Validation data.

        Returns
        -------
        AutoEncoder
            The instance with eval_loss updated.
        """
        self._validate_feature_names(X)
        dataset = OmicsDataset(X.values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.eval()
        eval_losses = []

        for data_batch in dataloader:
            data_batch = data_batch.to(self.device)
            data_batch_reconstructed = self.forward(data_batch, eval_mode=True)
            loss = self.criterion(data_batch_reconstructed, data_batch)
            eval_losses.append(loss.detach().cpu().numpy())

        self.eval_loss.append(np.mean(eval_losses))

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Encode the data using the fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        np.ndarray
            Transformed data in the learned representation space.
        """
        self._validate_feature_names(X)
        dataset = OmicsDataset(X.values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        features_list = []
        self.eval()

        with torch.no_grad():
            for data_batch in dataloader:
                data_batch = data_batch.to(self.device)
                data_batch_encoded = self.encoder(data_batch)
                features_list.append(data_batch_encoded.detach().cpu())

        features = torch.cat(features_list, dim=0).numpy()

        return features
