"""MLPRegressor using pytorch so that it can be accelerated with GPUs."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from leap.metrics.regression_metrics import performance_metric_wrapper

# RegressionModel is defined in __init__.py before this module is imported
from leap.regression_models import RegressionModel
from leap.utils.device import get_device
from leap.utils.seed import seed_everything

from .utils import SpearmanLoss


class TorchMLPRegressor(RegressionModel):
    """Multi-layer Perceptron Regressor implemented in PyTorch with GPU support.

    This implementation provides a flexible neural network regressor with support for early stopping, learning rate
    scheduling, dropout, and multiple loss functions.

    Parameters
    ----------
    hidden_layer_sizes : tuple
        The ith element represents the number of neurons in the ith hidden layer.
    activation : Literal["relu", "tanh"]
        Activation function for the hidden layers.
    learning_rate_init : float
        The initial learning rate for the optimizer.
    max_epochs : int
        Maximum number of epochs for training.
    batch_size : int
        Number of samples per batch for training.
    dropout_rate : float
        The dropout rate applied after each hidden layer. Must be in [0, 1).
    random_seed : int
        Random seed for reproducibility.
    early_stopping_use : bool
        Whether to use early stopping based on validation performance.
    early_stopping_split : float
        Fraction of training data to use for validation if early stopping is
        enabled and no validation set is provided. Must be in (0, 1).
    early_stopping_patience : int
        Number of epochs with no improvement after which training will be stopped.
    early_stopping_delta : float
        Minimum change in the monitored metric to qualify as an improvement.
    optimizer_type : Literal["adam", "sgd"]
        The optimizer to use for training.
    weight_decay : float
        Weight decay (L2 penalty) for the optimizer.
    learning_rate_scheduler : bool
        Whether to use a learning rate scheduler that reduces LR on plateau.
    scheduler_factor : float
        Factor by which the learning rate will be reduced.
    scheduler_patience : int
        Number of epochs with no improvement after which learning rate will be reduced.
    scheduler_threshold : float
        Threshold for measuring the new optimum for the scheduler.
    metric : Literal["spearman", "mse"]
        Metric function to evaluate the model during early stopping.
        Note: Despite the name, "spearman" actually uses Pearson correlation.
    scaler_name : Literal["standard", "minmax", "robust"] | None
        Scaler to use for feature normalization. If None, no scaling is applied.
    loss_function_name : Literal["mse", "spearman", "binary_cross_entropy"]
        Loss function to use for training.
        Note: "spearman" actually uses Pearson correlation.
    device : str | None
        Device to run training on ('cpu', 'cuda', or 'mps').
        If None, automatically detects best available device.

    Attributes
    ----------
    model : nn.Module
        The PyTorch neural network model.
    optimizer : optim.Optimizer
        The optimizer used for training.
    scheduler : optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler (if enabled).
    scaler : StandardScaler | MinMaxScaler | RobustScaler | None
        The fitted scaler for feature normalization.
    criterion : nn.Module
        The loss function module.
    loss_history_train : list[float]
        Training loss history per batch.
    loss_history_val : list[float]
        Validation loss history per epoch (if early stopping is used).
    metric_history_val : list[float]
        Validation metric history per epoch (if early stopping is used).


    Raises
    ------
    ValueError
        If the parameters are invalid.

    Notes
    -----
    - This implementation expects target data (y) to have a MultiIndex with a "perturbation" level when using early
    stopping.
    - The model is automatically moved to the appropriate device (CUDA, MPS, or CPU) based on availability.
    - BCEWithLogitsLoss includes sigmoid activation internally, so no sigmoid is added to the network when using binary
    cross-entropy loss.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from leap.regression_models import TorchMLPRegressor
    >>>
    >>> # Create sample data
    >>> X_train = pd.DataFrame(np.random.randn(100, 10))
    >>> y_train = pd.Series(np.random.randn(100))
    >>>
    >>> # Train model
    >>> model = TorchMLPRegressor(hidden_layer_sizes=(64, 32), max_epochs=50, early_stopping_use=False)
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Make predictions
    >>> X_test = pd.DataFrame(np.random.randn(20, 10))
    >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: Literal["relu", "tanh"] = "relu",
        learning_rate_init: float = 0.001,
        max_epochs: int = 200,
        batch_size: int = 64,
        dropout_rate: float = 0.0,
        random_seed: int = 0,
        early_stopping_use: bool = False,
        early_stopping_split: float = 0.2,
        early_stopping_patience: int = 20,
        early_stopping_delta: float = 0.0001,
        optimizer_type: Literal["adam", "sgd"] = "adam",
        weight_decay: float = 1e-5,
        learning_rate_scheduler: bool = False,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 10,
        scheduler_threshold: float = 0.001,
        metric: Literal["spearman", "mse"] = "spearman",
        scaler_name: Literal["standard", "minmax", "robust"] | None = "robust",
        loss_function_name: Literal["mse", "spearman", "binary_cross_entropy"] = "mse",
        device: str | None = None,
    ):
        # Validate parameters
        if not hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes must contain at least one layer")
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if not 0 < early_stopping_split < 1:
            raise ValueError(f"early_stopping_split must be in (0, 1), got {early_stopping_split}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Store the input parameters as instance attributes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.early_stopping_use = early_stopping_use
        self.early_stopping_split = early_stopping_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.learning_rate_scheduler = learning_rate_scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.n_epoch = 0
        self.loss_history_train: list[float] = []
        self.loss_history_val: list[float] = []
        self.metric_history_val: list[float] = []
        self.loss_function_name = loss_function_name
        self.scaler_name = scaler_name

        # Set the metric function
        self.metric = partial(performance_metric_wrapper, metric=metric, per_perturbation=True)
        self.metric_direction = -1 if metric == "mse" else 1

        # Set the random seeds for reproducibility
        seed_everything(self.random_seed)

        # Set device (CPU, CUDA, or MPS)
        self.device = get_device(device)

        # Placeholder for the PyTorch model and optimizer
        self.model: nn.Module
        self.optimizer: optim.Optimizer
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau

        # Scaler and loss function
        self.scaler, self.criterion = self._get_scaler_and_loss()

    def _get_scaler_and_loss(self) -> tuple[StandardScaler | MinMaxScaler | RobustScaler | None, nn.Module]:
        """Initialize the scaler and loss function.

        This becomes necessary as these are not args but need to be re-defined when set_params is called (used for CV).

        Returns
        -------
        tuple[StandardScaler | MinMaxScaler | RobustScaler | None, nn.Module]
            A tuple containing:
            - scaler: Optional sklearn scaler (StandardScaler, MinMaxScaler, RobustScaler, or None)
            - criterion: Loss function as an nn.Module (SpearmanLoss, BCEWithLogitsLoss, or MSELoss)

        Raises
        ------
        NotImplementedError
            If the loss function name is not supported.
        ValueError
            If the scaler name is not supported.
        """
        # Loss function
        criterion: nn.Module
        if self.loss_function_name == "spearman":
            criterion = SpearmanLoss()
        elif self.loss_function_name == "binary_cross_entropy":
            criterion = nn.BCEWithLogitsLoss()
        elif self.loss_function_name == "mse":
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss function '{self.loss_function_name}' is not supported or implemented.")

        # Scaler
        if self.scaler_name is None:
            scaler = None
        elif self.scaler_name == "standard":
            scaler = StandardScaler()
        elif self.scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif self.scaler_name == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler: {self.scaler_name}")

        return scaler, criterion

    def _create_dataloader(
        self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, device: str, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        """Create a PyTorch DataLoader from features and targets.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        y : pd.Series | pd.DataFrame
            Target data.
        device : str
            Device to load the tensors to ('cpu', 'cuda', or 'mps').
        batch_size : int
            Size of batches.
        shuffle : bool
            Whether to shuffle the data.

        Returns
        -------
        DataLoader
            PyTorch DataLoader containing the data.
        """
        return DataLoader(
            TensorDataset(
                torch.tensor(X.to_numpy(), dtype=torch.float32).to(device),
                torch.tensor(y.to_numpy(), dtype=torch.float32).to(device),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=False,
        )

    def _build_model(self, input_size: int, output_size: int) -> None:
        """Build the PyTorch model.

        Parameters
        ----------
        input_size : int
            Number of input features.
        output_size : int
            Number of output features.

        Raises
        ------
        ValueError
            If an unsupported activation function or optimizer type is provided.
        """
        layers: list[nn.Module] = []
        in_features = input_size

        # Select the activation function based on user input
        if self.activation == "relu":
            activation_fn: Callable[[], nn.Module] = nn.ReLU
        elif self.activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Add hidden layers with specified sizes and activation functions
        for hidden_size in self.hidden_layer_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    activation_fn(),
                    nn.Dropout(self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity(),
                ]
            )
            in_features = hidden_size

        # Add output layer
        layers.append(nn.Linear(in_features, output_size))

        # Create sequential model from layers
        self.model = nn.Sequential(*layers)

        # Select optimizer
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Set up learning rate scheduler, if enabled
        if self.learning_rate_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min" if self.metric_direction == -1 else "max",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold,
            )

    def _get_val_tensors(
        self, X_val: pd.DataFrame, y_val: pd.Series | pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, pd.Index]:
        """Get validation tensors.

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation features.
        y_val : pd.Series | pd.DataFrame
            Validation targets. Must have a MultiIndex with a "perturbation" level.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, pd.Index]
            Validation features tensor, validation targets tensor, and perturbations index.

        Raises
        ------
        KeyError
            If y_val index doesn't have a "perturbation" level.
        """
        # Convert validation data to PyTorch tensors
        X_val_array = self.scaler.transform(X_val) if self.scaler is not None else X_val.to_numpy()
        X_val_tensor = torch.tensor(X_val_array, dtype=torch.float32).to(self.device)

        try:
            val_perturbations = y_val.index.get_level_values("perturbation")
        except KeyError as e:
            raise KeyError(
                "y_val must have a MultiIndex with a 'perturbation' level for early stopping. "
                f"Got index levels: {y_val.index.names}"
            ) from e

        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).to(self.device)
        return X_val_tensor, y_val_tensor, val_perturbations

    def _get_train_val_split(
        self, X: pd.DataFrame, y: pd.Series | pd.DataFrame
    ) -> tuple[DataLoader, torch.Tensor, torch.Tensor, pd.Index]:
        """Create training and validation dataloaders.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        y : pd.Series | pd.DataFrame
            Target data. Must have a MultiIndex with a "perturbation" level.

        Returns
        -------
        tuple[DataLoader, torch.Tensor, torch.Tensor, pd.Index]
            Training dataloader, validation features tensor, validation targets tensor, and validation perturbations
            index.

        Raises
        ------
        KeyError
            If y index doesn't have a "perturbation" level.
        """
        # Convert input and target data to PyTorch tensors
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).to(self.device)

        # Split the data into training and validation sets
        dataset_size = len(X_tensor)
        val_size = int(self.early_stopping_split * dataset_size)
        train_size = dataset_size - val_size
        indices = torch.tensor(range(dataset_size))
        train_dataset, val_dataset = random_split(
            TensorDataset(X_tensor, y_tensor, indices),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False)

        # Extract X_val and y_val from val_dataset
        # For validation data, we don't need a DataLoader since:
        # 1. All data fits in memory
        # 2. We don't need batching since we're not training
        # 3. We evaluate on the full validation set at once
        X_val = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
        y_val = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])
        val_indices = torch.stack([val_dataset[i][2] for i in range(len(val_dataset))])

        try:
            val_perturbations = y.iloc[val_indices.cpu().numpy()].index.get_level_values("perturbation")
        except KeyError as e:
            raise KeyError(
                "y must have a MultiIndex with a 'perturbation' level for early stopping. "
                f"Got index levels: {y.index.names}"
            ) from e

        # Clean GPU memory
        del X_tensor, y_tensor, train_dataset, val_dataset, val_indices

        return train_dataloader, X_val, y_val, val_perturbations

    def _early_stopping(
        self,
        epoch: int,
        model_gpu: nn.Module,
        patience_counter: int,
        best_metric: float,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        val_perturbations: pd.Index,
    ) -> tuple[int, float, float, dict | None]:
        """Early stopping logic.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        model_gpu : nn.Module
            Model on GPU.
        patience_counter : int
            Current patience counter.
        best_metric : float
            Best metric achieved so far.
        X_val : torch.Tensor
            Validation features.
        y_val : torch.Tensor
            Validation targets.
        val_perturbations : pd.Index
            Validation perturbations index.

        Returns
        -------
        tuple[int, float, float, dict | None]
            Updated patience counter, validation metric, best metric, and best model state dict.
        """
        # Create placeholder for the best model state
        best_model_state: dict | None = None

        # Validation phase
        model_gpu.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            # Use no dataloader to avoid extra overhead when doing per-perturbation predictions and metrics
            # + all data fits in memory
            val_predictions = model_gpu(X_val)  # Forward pass
            val_targets = y_val.view(-1, 1)  # Reshape target to match output
            val_loss = self.criterion(val_predictions, val_targets)  # Loss

        # Compute validation metric per perturbation
        val_targets_series = pd.Series(
            val_targets.cpu().numpy().flatten(),
            index=pd.MultiIndex.from_arrays(
                [range(len(val_targets)), val_perturbations], names=["sample", "perturbation"]
            ),
        )
        val_predictions_series = pd.Series(
            val_predictions.cpu().numpy().flatten(),
            index=pd.MultiIndex.from_arrays(
                [range(len(val_targets)), val_perturbations], names=["sample", "perturbation"]
            ),
        )

        val_metric = self.metric(val_targets_series, val_predictions_series)
        self.loss_history_val.append(val_loss.item())
        self.metric_history_val.append(val_metric)

        # Early stopping logic
        val_metric *= self.metric_direction
        if val_metric > best_metric + self.early_stopping_delta:
            best_metric = val_metric
            patience_counter = 0
            best_model_state = model_gpu.state_dict()
        else:
            patience_counter += 1

        return patience_counter, val_metric, best_metric, best_model_state

    def fit(  # noqa: PLR0912, PLR0915
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the model to the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data features.
        y : pd.Series | pd.DataFrame
            Training data target values. If early_stopping_use is True, must have a MultiIndex with a "perturbation"
            level.
        X_val : pd.DataFrame | None
            Validation data features for early stopping. If early_stopping_use is True and this is None, a validation
            split will be created from the training data.
        y_val : pd.Series | pd.DataFrame | None
            Validation data target values for early stopping. If early_stopping_use is True and this is None, a
            validation split will be created from the training data. Must have a MultiIndex with a "perturbation" level.

        Raises
        ------
        ValueError
            If early_stopping_use is True and only one of X_val or y_val is provided (both must be provided together or
            neither).
        RuntimeError
            If an internal consistency check fails during training.
        """
        # Set the random seeds for reproducibility
        seed_everything(self.random_seed)

        # Scale features (create a copy to avoid mutating input)
        X_scaled = pd.DataFrame(
            (self.scaler.fit_transform(X) if self.scaler is not None else X.to_numpy()),
            index=X.index,
            columns=X.columns,
        )

        # Build the model if not already built
        if not hasattr(self, "model"):
            # Determine input and output sizes
            input_size = X_scaled.shape[1]
            output_size = y.shape[1] if len(y.shape) > 1 else 1
            self._build_model(input_size, output_size)

        # Move model to appropriate device (CPU or GPU)
        model_gpu = self.model.to(self.device)

        # PREPARE DATA
        if self.early_stopping_use and X_val is None and y_val is None:
            train_dataloader, X_val_tensor, y_val_tensor, val_perturbations = self._get_train_val_split(X_scaled, y)
        else:
            train_dataloader = self._create_dataloader(
                X=X_scaled, y=y, device=self.device, batch_size=self.batch_size, shuffle=True
            )
            if self.early_stopping_use:
                # Validate that both X_val and y_val are provided together
                if X_val is None or y_val is None:
                    raise ValueError(
                        "When early_stopping_use is True and validation data is provided, "
                        "both X_val and y_val must be provided together. "
                        f"Got X_val={'provided' if X_val is not None else 'None'}, "
                        f"y_val={'provided' if y_val is not None else 'None'}."
                    )
                X_val_tensor, y_val_tensor, val_perturbations = self._get_val_tensors(X_val, y_val)
            else:
                val_perturbations = None

        # Initialize best metric for early stopping
        if self.early_stopping_use:
            best_metric = -np.inf
            patience_counter = 0
            best_model_state: dict | None = None

        # TRAINING LOOP
        for epoch in range(self.max_epochs):
            self.n_epoch = epoch
            # Training phase
            model_gpu.train()  # Set model to training mode
            train_losses = []  # Initialize list to track training losses

            for batch in train_dataloader:
                X_batch, y_batch = batch[0], batch[1]
                self.optimizer.zero_grad()  # Zero the gradients
                outputs = model_gpu(X_batch)  # Forward pass
                y_batch = y_batch.view(-1, 1)  # Reshape target to match output
                loss = self.criterion(outputs, y_batch)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                train_losses.append(loss.item())  # Append current batch loss
                self.loss_history_train.append(loss.item())

            # Handle early stopping
            if self.early_stopping_use:
                # Internal consistency check
                if val_perturbations is None:
                    raise RuntimeError(
                        "Internal error: val_perturbations is None despite early_stopping_use=True. "
                        "This indicates a bug in the data preparation logic."
                    )
                patience_counter, val_metric, best_metric, best_model_state = self._early_stopping(
                    epoch=epoch,
                    model_gpu=model_gpu,
                    patience_counter=patience_counter,
                    best_metric=best_metric,
                    X_val=X_val_tensor,
                    y_val=y_val_tensor,
                    val_perturbations=val_perturbations,
                )

                # Learning rate scheduling (when early stopping is used)
                if self.learning_rate_scheduler:
                    self.scheduler.step(metrics=val_metric)

                # Check if patience exceeded
                if patience_counter >= self.early_stopping_patience:
                    break
            elif self.learning_rate_scheduler:
                # Learning rate scheduling without early stopping
                # Use training loss as metric
                self.scheduler.step(metrics=np.mean(train_losses))

        # Load the best model state if available
        if self.early_stopping_use and best_model_state is not None:
            model_gpu.load_state_dict(best_model_state)

        # Move model back to CPU for prediction consistency
        self.model = model_gpu.cpu()

        # Clean up memory
        if self.early_stopping_use:
            del X_val_tensor, y_val_tensor, best_model_state
        del model_gpu, train_dataloader, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        np.ndarray
            Predictions as a 1D array.
        """
        # Scale input data
        X_array = self.scaler.transform(X) if self.scaler is not None else X.to_numpy()

        # Convert input data to PyTorch tensor
        # Model is on CPU after training, so put data on CPU too
        X_tensor = torch.tensor(X_array, dtype=torch.float32)

        # Set model to evaluation mode
        self.model.eval()

        # Disable gradient computation for prediction
        with torch.no_grad():
            predictions = self.model(X_tensor)

        # Return predictions as a NumPy array
        return predictions.view(-1).cpu().numpy()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "learning_rate_init": self.learning_rate_init,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "dropout_rate": self.dropout_rate,
            "random_seed": self.random_seed,
            "early_stopping_use": self.early_stopping_use,
            "early_stopping_split": self.early_stopping_split,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_delta": self.early_stopping_delta,
            "optimizer_type": self.optimizer_type,
            "weight_decay": self.weight_decay,
            "learning_rate_scheduler": self.learning_rate_scheduler,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_threshold": self.scheduler_threshold,
            "metric": self.metric,
            "scaler_name": self.scaler_name,
            "loss_function_name": self.loss_function_name,
        }

    def set_params(self, **kwargs: Any) -> TorchMLPRegressor:
        """Set parameters for this estimator."""
        for parameter, value in kwargs.items():
            setattr(self, parameter, value)
        self.scaler, self.criterion = self._get_scaler_and_loss()
        return self

    def __del__(self) -> None:
        """Cleanup method to free GPU memory when the object is destroyed."""
        if hasattr(self, "model"):
            del self.model
        if "torch" in globals() and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "torch" in globals() and getattr(torch, "mps", None) is not None and torch.mps.is_available():
            torch.mps.empty_cache()
