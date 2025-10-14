"""Tests for the regression_models module."""

import numpy as np
import pandas as pd
import pytest
import torch

from leap.regression_models import ElasticNet, KnnRegressor, LGBMRegressor, RegressionModel, TorchMLPRegressor
from leap.regression_models.utils import SpearmanLoss


class TestRegressionModelABC:
    """Test RegressionModel abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that RegressionModel ABC cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RegressionModel()

    def test_all_models_have_required_methods(self):
        """Test that all regression models implement required methods."""
        required_methods = ["fit", "predict", "set_params", "get_params"]

        models = [
            ElasticNet(alpha=1.0, l1_ratio=0.5),
            LGBMRegressor(n_estimators=10, verbose=-1),
            TorchMLPRegressor(hidden_layer_sizes=(10,), max_epochs=5),
            KnnRegressor(n_sample_neighbors=5, weights="uniform"),
        ]

        for model in models:
            for method in required_methods:
                assert hasattr(model, method), f"{type(model).__name__} missing {method} method"
                assert callable(getattr(model, method)), f"{type(model).__name__}.{method} is not callable"


class TestElasticNetWrapper:
    """Test ElasticNet wrapper class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)])
        y = pd.Series(np.random.randn(100))
        return X, y

    def test_elastic_net_initialization(self):
        """Test ElasticNet initialization."""
        model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        assert model.l1_ratio == 0.5

    def test_elastic_net_fit(self, sample_data):
        """Test ElasticNet fit method with pandas DataFrames."""
        X, y = sample_data
        model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        model.fit(X, y)
        # Should not raise an error

    def test_elastic_net_predict(self, sample_data):
        """Test ElasticNet predict method."""
        X, y = sample_data
        model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        model.fit(X, y)

        X_test = X[:10]
        predictions = model.predict(X_test)

        assert predictions.shape == (10,)
        assert isinstance(predictions, np.ndarray)

    def test_elastic_net_get_set_params(self):
        """Test get_params and set_params methods."""
        model = ElasticNet(alpha=1.0, l1_ratio=0.5)

        params = model.get_params()
        assert params["alpha"] == 1.0
        assert params["l1_ratio"] == 0.5

        model.set_params(alpha=2.0)
        assert model.get_params()["alpha"] == 2.0

    def test_elastic_net_with_series_and_dataframe(self, sample_data):
        """Test ElasticNet works with both Series and DataFrame targets."""
        X, y = sample_data
        model = ElasticNet(alpha=1.0, l1_ratio=0.5)

        # Test with Series
        model.fit(X, y)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5,)

        # Test with DataFrame (single column)
        y_df = pd.DataFrame(y, columns=["target"])
        model.fit(X, y_df)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5,)


class TestLGBMRegressorWrapper:
    """Test LGBMRegressor wrapper class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)])
        y = pd.Series(np.random.randn(100))
        return X, y

    def test_lgbm_initialization(self):
        """Test LGBMRegressor initialization."""
        LGBMRegressor(n_estimators=10, verbose=-1)
        # Should initialize without error

    def test_lgbm_fit(self, sample_data):
        """Test LGBMRegressor fit method with pandas DataFrames."""
        X, y = sample_data
        model = LGBMRegressor(n_estimators=10, verbose=-1)
        model.fit(X, y)
        # Should not raise an error

    def test_lgbm_predict(self, sample_data):
        """Test LGBMRegressor predict method."""
        X, y = sample_data
        model = LGBMRegressor(n_estimators=10, verbose=-1)
        model.fit(X, y)

        X_test = X[:10]
        predictions = model.predict(X_test)

        assert predictions.shape == (10,)
        assert isinstance(predictions, np.ndarray)

    def test_lgbm_get_set_params(self):
        """Test get_params and set_params methods."""
        model = LGBMRegressor(n_estimators=10, verbose=-1)

        params = model.get_params()
        assert params["n_estimators"] == 10
        assert params["verbose"] == -1

        model.set_params(n_estimators=20)
        assert model.get_params()["n_estimators"] == 20

    def test_lgbm_with_series_and_dataframe(self, sample_data):
        """Test LGBMRegressor works with both Series and DataFrame targets."""
        X, y = sample_data
        model = LGBMRegressor(n_estimators=10, verbose=-1)

        # Test with Series
        model.fit(X, y)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5,)

        # Test with DataFrame (single column)
        y_df = pd.DataFrame(y, columns=["target"])
        model.fit(X, y_df)
        predictions = model.predict(X[:5])
        assert predictions.shape == (5,)


class TestKnnRegressor:
    """Test KnnRegressor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)])
        # Multiple targets with some NaN values
        y = pd.DataFrame(
            {
                "target1": np.random.randn(100),
                "target2": np.random.randn(100),
                "target3": np.random.randn(100),
            }
        )
        # Add some NaN values
        y.iloc[0:10, 0] = np.nan
        y.iloc[20:30, 1] = np.nan

        return X, y

    def test_knn_initialization(self):
        """Test KNN initialization."""
        knn = KnnRegressor(n_sample_neighbors=5, weights="uniform")

        assert knn.n_sample_neighbors == 5
        assert knn.weights == "uniform"

    def test_knn_fit(self, sample_data):
        """Test KNN fit method."""
        X, y = sample_data
        knn = KnnRegressor(n_sample_neighbors=5, weights="uniform")

        knn.fit(X, y)

        # Should store training data
        assert hasattr(knn, "X_train")
        assert hasattr(knn, "y_train")
        pd.testing.assert_frame_equal(knn.X_train, X)
        pd.testing.assert_frame_equal(knn.y_train, y)

    def test_knn_predict(self, sample_data):
        """Test KNN predict method."""
        X, y = sample_data
        knn = KnnRegressor(n_sample_neighbors=5, weights="uniform")
        knn.fit(X, y)

        # Predict on new data
        X_test = X[:10]
        predictions = knn.predict(X_test)

        assert predictions.shape == (10, 3)  # 10 samples, 3 targets

    def test_knn_handles_nan_values(self, sample_data):
        """Test that KNN properly handles NaN values in targets."""
        X, y = sample_data
        knn = KnnRegressor(n_sample_neighbors=5, weights="uniform")
        knn.fit(X, y)

        # Should still work despite NaN values
        X_test = X[:5]
        predictions = knn.predict(X_test)

        # Predictions should not contain NaN
        assert not np.isnan(predictions).any()

    def test_knn_distance_weighted(self, sample_data):
        """Test KNN with distance weighting."""
        X, y = sample_data
        knn = KnnRegressor(n_sample_neighbors=5, weights="distance")
        knn.fit(X, y)

        X_test = X[:5]
        predictions = knn.predict(X_test)

        assert predictions.shape == (5, 3)

    def test_knn_fewer_neighbors_than_samples(self, sample_data):
        """Test KNN when n_neighbors > n_samples for some target."""
        X, y = sample_data
        # Use large number of neighbors
        knn = KnnRegressor(n_sample_neighbors=200, weights="uniform")
        knn.fit(X, y)

        # Should still work by using min(n_neighbors, n_available_samples)
        X_test = X[:5]
        predictions = knn.predict(X_test)

        assert predictions.shape == (5, 3)


class TestTorchMLPRegressor:
    """Test TorchMLPRegressor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)])
        y = pd.Series(np.random.randn(100))

        return X, y

    def test_mlp_initialization(self):
        """Test MLP initialization."""
        mlp = TorchMLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_epochs=10)

        assert mlp.hidden_layer_sizes == (64, 32)
        assert mlp.activation == "relu"
        assert mlp.max_epochs == 10

    def test_mlp_invalid_hidden_layers_raises_error(self):
        """Test that empty hidden layers raises error."""
        with pytest.raises(ValueError, match="hidden_layer_sizes must contain at least one layer"):
            TorchMLPRegressor(hidden_layer_sizes=())

    def test_mlp_invalid_dropout_raises_error(self):
        """Test that invalid dropout raises error."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            TorchMLPRegressor(dropout_rate=1.5)

    def test_mlp_invalid_early_stopping_split_raises_error(self):
        """Test that invalid early stopping split raises error."""
        with pytest.raises(ValueError, match="early_stopping_split must be in"):
            TorchMLPRegressor(early_stopping_split=1.5)

    def test_mlp_fit(self, sample_data):
        """Test MLP fit method."""
        X, y = sample_data
        mlp = TorchMLPRegressor(hidden_layer_sizes=(32,), max_epochs=5, early_stopping_use=False)

        mlp.fit(X, y)

        assert hasattr(mlp, "model")
        assert len(mlp.loss_history_train) > 0

    def test_mlp_predict(self, sample_data):
        """Test MLP predict method."""
        X, y = sample_data
        mlp = TorchMLPRegressor(hidden_layer_sizes=(32,), max_epochs=5, early_stopping_use=False)
        mlp.fit(X, y)

        X_test = X[:10]
        predictions = mlp.predict(X_test)

        assert predictions.shape == (10,)
        assert isinstance(predictions, np.ndarray)

    def test_mlp_with_early_stopping(self, sample_data):
        """Test MLP with early stopping."""
        X, y = sample_data

        # Add perturbation index for early stopping
        y.index = pd.MultiIndex.from_tuples(
            [(i, "perturbation") for i in range(len(y))], names=["sample", "perturbation"]
        )

        mlp = TorchMLPRegressor(
            hidden_layer_sizes=(32,),
            max_epochs=50,
            early_stopping_use=True,
            early_stopping_patience=5,
            early_stopping_split=0.2,
        )
        mlp.fit(X, y)

        # Should have validation metrics
        assert len(mlp.loss_history_val) > 0
        assert len(mlp.metric_history_val) > 0

    def test_mlp_with_external_validation(self, sample_data):
        """Test MLP with external validation data."""
        X, y = sample_data

        # Add perturbation index
        y.index = pd.MultiIndex.from_tuples(
            [(i, "perturbation") for i in range(len(y))], names=["sample", "perturbation"]
        )

        # Split data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        mlp = TorchMLPRegressor(
            hidden_layer_sizes=(32,), max_epochs=10, early_stopping_use=True, early_stopping_patience=5
        )
        mlp.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        assert len(mlp.loss_history_val) > 0

    def test_mlp_scalers(self, sample_data):
        """Test different scalers."""
        X, y = sample_data

        for scaler_name in ["standard", "minmax", "robust", None]:
            mlp = TorchMLPRegressor(
                hidden_layer_sizes=(16,), max_epochs=3, early_stopping_use=False, scaler_name=scaler_name
            )
            mlp.fit(X, y)
            predictions = mlp.predict(X[:5])
            assert predictions.shape == (5,)

    def test_mlp_loss_functions(self, sample_data):
        """Test different loss functions."""
        X, y = sample_data

        for loss_name in ["mse", "spearman"]:
            mlp = TorchMLPRegressor(
                hidden_layer_sizes=(16,), max_epochs=3, early_stopping_use=False, loss_function_name=loss_name
            )
            mlp.fit(X, y)
            predictions = mlp.predict(X[:5])
            assert predictions.shape == (5,)

    def test_mlp_get_set_params(self):
        """Test get_params and set_params methods."""
        mlp = TorchMLPRegressor(hidden_layer_sizes=(64, 32), learning_rate_init=0.001)

        params = mlp.get_params()
        assert params["hidden_layer_sizes"] == (64, 32)
        assert params["learning_rate_init"] == 0.001

        mlp.set_params(learning_rate_init=0.01)
        assert mlp.learning_rate_init == 0.01

    def test_mlp_dropout(self, sample_data):
        """Test MLP with dropout."""
        X, y = sample_data
        mlp = TorchMLPRegressor(hidden_layer_sizes=(32, 16), dropout_rate=0.3, max_epochs=5, early_stopping_use=False)

        mlp.fit(X, y)
        predictions = mlp.predict(X[:10])

        assert predictions.shape == (10,)

    def test_mlp_learning_rate_scheduler(self, sample_data):
        """Test MLP with learning rate scheduler."""
        X, y = sample_data

        # Add perturbation index for early stopping
        y.index = pd.MultiIndex.from_tuples(
            [(i, "perturbation") for i in range(len(y))], names=["sample", "perturbation"]
        )

        mlp = TorchMLPRegressor(
            hidden_layer_sizes=(32,),
            max_epochs=20,
            early_stopping_use=True,
            early_stopping_patience=10,
            learning_rate_scheduler=True,
            scheduler_patience=5,
        )
        mlp.fit(X, y)

        # Should have trained successfully
        assert hasattr(mlp, "scheduler")


class TestSpearmanLoss:
    """Test SpearmanLoss class."""

    def test_spearman_loss_perfect_correlation(self):
        """Test SpearmanLoss with perfect correlation."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)

        # Loss should be close to 0 (1 - 1 = 0)
        assert loss.item() < 0.01

    def test_spearman_loss_negative_correlation(self):
        """Test SpearmanLoss with negative correlation."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)

        # Loss should be close to 2 (1 - (-1) = 2)
        assert loss.item() > 1.9

    def test_spearman_loss_no_correlation(self):
        """Test SpearmanLoss with no correlation."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([3.0, 1.0, 4.0, 2.0, 5.0])

        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)

        # Loss should be around 1 (correlation near 0)
        assert 0.4 < loss.item() < 1.5

    def test_spearman_loss_gradient(self):
        """Test that SpearmanLoss produces gradients."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([1.1, 2.2, 2.9, 4.1, 5.2], requires_grad=True)

        loss_fn = SpearmanLoss()
        loss = loss_fn(y_pred, y_true)
        loss.backward()

        # Should have gradients
        assert y_pred.grad is not None
        assert not torch.all(y_pred.grad == 0)
