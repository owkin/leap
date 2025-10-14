"""Tests for the representation_models module."""

import numpy as np
import pandas as pd
import pytest
import torch

from leap.representation_models import PCA, AutoEncoder, MaskedAutoencoder, RepresentationModelBase
from leap.representation_models.utils import OmicsDataset, _initialize_early_stopping, _update_early_stopping


class TestPCA:
    """Test PCA representation model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(100, 50), columns=[f"feature_{i}" for i in range(50)])
        return data

    def test_pca_initialization(self):
        """Test PCA initialization."""
        pca = PCA(repr_dim=10, random_state=42)

        assert pca.repr_dim == 10
        assert pca.n_components == 10

    def test_pca_is_representation_model(self):
        """Test that PCA implements RepresentationModelBase."""
        pca = PCA(repr_dim=10)
        assert isinstance(pca, RepresentationModelBase)

    def test_pca_fit(self, sample_data):
        """Test PCA fit method."""
        pca = PCA(repr_dim=10, random_state=42)
        pca.fit(sample_data)

        assert hasattr(pca, "components_")
        assert pca.components_.shape[0] == 10

    def test_pca_transform(self, sample_data):
        """Test PCA transform method."""
        pca = PCA(repr_dim=10, random_state=42)
        pca.fit(sample_data)

        transformed = pca.transform(sample_data)

        assert transformed.shape == (100, 10)
        assert isinstance(transformed, np.ndarray)

    def test_pca_reproducibility(self, sample_data):
        """Test that PCA produces reproducible results."""
        pca1 = PCA(repr_dim=10, random_state=42)
        pca1.fit(sample_data)
        result1 = pca1.transform(sample_data)

        pca2 = PCA(repr_dim=10, random_state=42)
        pca2.fit(sample_data)
        result2 = pca2.transform(sample_data)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_pca_variance_explained(self, sample_data):
        """Test that PCA components explain variance."""
        pca = PCA(repr_dim=10, random_state=42)
        pca.fit(sample_data)

        # Check that explained variance exists and sums to reasonable value
        assert hasattr(pca, "explained_variance_ratio_")
        assert len(pca.explained_variance_ratio_) == 10
        assert pca.explained_variance_ratio_.sum() > 0


class TestAutoEncoder:
    """Test AutoEncoder representation model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(50, 30), columns=[f"feature_{i}" for i in range(30)])
        return data

    def test_autoencoder_initialization(self):
        """Test AutoEncoder initialization."""
        ae = AutoEncoder(
            repr_dim=10, hidden_n_layers=2, hidden_n_units_first=20, num_epochs=5, early_stopping_use=False
        )

        assert ae.repr_dim == 10
        assert ae.hidden_n_layers == 2
        assert ae.num_epochs == 5

    def test_autoencoder_is_representation_model(self):
        """Test that AutoEncoder implements RepresentationModelBase."""
        ae = AutoEncoder(repr_dim=10, num_epochs=5, early_stopping_use=False)
        assert isinstance(ae, RepresentationModelBase)

    def test_autoencoder_hidden_config_conversion(self):
        """Test hidden layer configuration conversion."""
        ae = AutoEncoder(
            repr_dim=10,
            hidden_n_layers=3,
            hidden_n_units_first=100,
            hidden_decrease_rate=0.5,
            num_epochs=5,
            early_stopping_use=False,
        )

        expected_hidden = [100, 50, 25]
        assert ae.hidden == expected_hidden

    def test_autoencoder_fit(self, sample_data):
        """Test AutoEncoder fit method."""
        ae = AutoEncoder(
            repr_dim=5,
            hidden_n_layers=1,
            hidden_n_units_first=10,
            num_epochs=3,
            batch_size=16,
            early_stopping_use=False,
        )

        ae.fit(sample_data)

        assert hasattr(ae, "encoder")
        assert hasattr(ae, "decoder")
        assert len(ae.train_loss) > 0

    def test_autoencoder_transform(self, sample_data):
        """Test AutoEncoder transform method."""
        ae = AutoEncoder(repr_dim=5, num_epochs=3, early_stopping_use=False)

        ae.fit(sample_data)
        transformed = ae.transform(sample_data)

        assert transformed.shape == (50, 5)
        assert isinstance(transformed, np.ndarray)

    def test_autoencoder_early_stopping(self, sample_data):
        """Test AutoEncoder with early stopping."""
        ae = AutoEncoder(
            repr_dim=5,
            max_num_epochs=100,
            early_stopping_use=True,
            early_stopping_patience=5,
            early_stopping_split=0.2,
            retrain=False,  # Don't retrain for faster test
        )

        ae.fit(sample_data)

        # Should have stopped early
        assert ae.early_stopping_epoch < 100
        assert len(ae.eval_loss) > 0

    def test_autoencoder_feature_name_validation(self, sample_data):
        """Test feature name validation during transform."""
        ae = AutoEncoder(repr_dim=5, num_epochs=3, early_stopping_use=False)
        ae.fit(sample_data)

        # Try to transform data with different features
        wrong_data = pd.DataFrame(np.random.randn(10, 30), columns=[f"wrong_feature_{i}" for i in range(30)])

        with pytest.raises(ValueError, match="feature names should match"):
            ae.transform(wrong_data)

    def test_autoencoder_device_handling(self, sample_data):
        """Test that AutoEncoder handles device correctly."""
        ae = AutoEncoder(repr_dim=5, num_epochs=3, early_stopping_use=False, device="cpu")

        ae.fit(sample_data)
        assert ae.device == "cpu"

    def test_autoencoder_forward_pass(self, sample_data):
        """Test AutoEncoder forward pass."""
        ae = AutoEncoder(repr_dim=5, num_epochs=3, early_stopping_use=False, device="cpu")
        ae.fit(sample_data)

        # Test forward pass
        sample_tensor = torch.tensor(sample_data.values[:5], dtype=torch.float32)
        reconstructed = ae.forward(sample_tensor)

        assert reconstructed.shape == sample_tensor.shape


class TestMaskedAutoencoder:
    """Test MaskedAutoencoder representation model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(50, 30), columns=[f"feature_{i}" for i in range(30)])
        return data

    def test_masked_autoencoder_initialization(self):
        """Test MaskedAutoencoder initialization."""
        mae = MaskedAutoencoder(
            repr_dim=10, corruption_proba=0.3, corruption_method="classic", num_epochs=5, early_stopping_use=False
        )

        assert mae.repr_dim == 10
        assert mae.corruption_proba == 0.3
        assert mae.corruption_method == "classic"

    def test_masked_autoencoder_invalid_corruption_method(self):
        """Test that invalid corruption method raises error."""
        with pytest.raises(ValueError, match="corruption_method must be one of"):
            MaskedAutoencoder(repr_dim=10, corruption_method="invalid_method", num_epochs=5, early_stopping_use=False)

    def test_masked_autoencoder_mask_generator(self, sample_data):
        """Test mask generation."""
        mae = MaskedAutoencoder(repr_dim=5, corruption_proba=0.3, num_epochs=1, early_stopping_use=False)

        sample_tensor = torch.tensor(sample_data.values[:10], dtype=torch.float32)
        mask = mae.mask_generator(sample_tensor)

        assert mask.shape == sample_tensor.shape
        # Approximately 30% should be masked
        mask_ratio = mask.mean().item()
        assert 0.1 < mask_ratio < 0.5

    def test_masked_autoencoder_classic_corruption(self, sample_data):
        """Test classic masking (zero corruption)."""
        mae = MaskedAutoencoder(
            repr_dim=5,
            corruption_method="classic",
            corruption_proba=1.0,  # Mask everything for testing
            num_epochs=1,
            early_stopping_use=False,
            device="cpu",
        )

        sample_tensor = torch.tensor(sample_data.values[:10], dtype=torch.float32)
        mask = torch.ones_like(sample_tensor)
        corrupted = mae.pretext_generator(mask, sample_tensor)

        # With full masking and classic method, should be all zeros
        assert corrupted.sum().item() == 0

    def test_masked_autoencoder_noise_corruption(self, sample_data):
        """Test noise corruption."""
        mae = MaskedAutoencoder(repr_dim=5, corruption_method="noise", beta=0.1, num_epochs=1, early_stopping_use=False)

        sample_tensor = torch.tensor(sample_data.values[:10], dtype=torch.float32)
        mask = torch.ones_like(sample_tensor)
        corrupted = mae.pretext_generator(mask, sample_tensor)

        # With noise, corrupted should be different from original
        assert not torch.allclose(corrupted, sample_tensor)

    def test_masked_autoencoder_vime_corruption(self, sample_data):
        """Test VIME corruption (permutation)."""
        mae = MaskedAutoencoder(repr_dim=5, corruption_method="vime", num_epochs=1, early_stopping_use=False)

        sample_tensor = torch.tensor(sample_data.values[:10], dtype=torch.float32)
        mask = torch.ones_like(sample_tensor)
        corrupted = mae.pretext_generator(mask, sample_tensor)

        # Shape should be the same
        assert corrupted.shape == sample_tensor.shape

    def test_masked_autoencoder_fit_transform(self, sample_data):
        """Test MaskedAutoencoder fit and transform."""
        mae = MaskedAutoencoder(
            repr_dim=5, corruption_proba=0.3, corruption_method="classic", num_epochs=3, early_stopping_use=False
        )

        mae.fit(sample_data)
        transformed = mae.transform(sample_data)

        assert transformed.shape == (50, 5)
        assert isinstance(transformed, np.ndarray)

    def test_masked_autoencoder_forward_eval_mode(self, sample_data):
        """Test forward pass in eval mode (no masking)."""
        mae = MaskedAutoencoder(repr_dim=5, corruption_proba=0.3, num_epochs=3, early_stopping_use=False, device="cpu")
        mae.fit(sample_data)

        sample_tensor = torch.tensor(sample_data.values[:5], dtype=torch.float32)

        # In eval mode, should just do standard autoencoding
        mae.eval()
        reconstructed = mae.forward(sample_tensor, eval_mode=True)

        assert reconstructed.shape == sample_tensor.shape

    def test_masked_autoencoder_data_augmentation(self, sample_data):
        """Test data augmentation."""
        mae = MaskedAutoencoder(
            repr_dim=5,
            corruption_method="classic",
            data_augmentation=True,
            da_noise_std=0.01,
            num_epochs=3,
            early_stopping_use=False,
        )

        mae.fit(sample_data)
        transformed = mae.transform(sample_data)

        assert transformed.shape == (50, 5)


class TestRepresentationUtils:
    """Test utility functions for representation models."""

    def test_omics_dataset(self):
        """Test OmicsDataset class."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        dataset = OmicsDataset(X, y)

        assert len(dataset) == 10
        features, label = dataset[0]
        assert features.shape == (5,)
        assert label.shape == (1,)

    def test_omics_dataset_without_labels(self):
        """Test OmicsDataset without labels."""
        X = np.random.randn(10, 5)

        dataset = OmicsDataset(X, y=None)

        assert len(dataset) == 10
        features = dataset[0]
        assert isinstance(features, torch.Tensor)
        assert features.shape == (5,)

    def test_initialize_early_stopping_first_epoch(self):
        """Test early stopping initialization at first epoch."""
        eval_loss = [0.5]
        early_stopping_best = 0.0

        result = _initialize_early_stopping(eval_loss, early_stopping_best)

        assert result == 0.5

    def test_initialize_early_stopping_later_epoch(self):
        """Test early stopping initialization after first epoch."""
        eval_loss = [0.5, 0.4]
        early_stopping_best = 0.5

        result = _initialize_early_stopping(eval_loss, early_stopping_best)

        assert result == 0.5

    def test_update_early_stopping_improvement_loss(self):
        """Test early stopping update with improvement (lower loss)."""
        eval_list = [0.5, 0.4, 0.3]
        early_stopping_best = 0.4
        early_stopping_delta = 0.01
        patience_count = 2

        best, patience = _update_early_stopping(
            eval_list, early_stopping_best, early_stopping_delta, patience_count, use_metric=False
        )

        assert best == 0.3
        assert patience == 0

    def test_update_early_stopping_no_improvement_loss(self):
        """Test early stopping update without improvement (loss)."""
        eval_list = [0.5, 0.4, 0.41]
        early_stopping_best = 0.4
        early_stopping_delta = 0.01
        patience_count = 0

        best, patience = _update_early_stopping(
            eval_list, early_stopping_best, early_stopping_delta, patience_count, use_metric=False
        )

        assert best == 0.4
        assert patience == 1

    def test_update_early_stopping_improvement_metric(self):
        """Test early stopping update with improvement (higher metric)."""
        eval_list = [0.5, 0.6, 0.7]
        early_stopping_best = 0.6
        early_stopping_delta = 0.01
        patience_count = 2

        best, patience = _update_early_stopping(
            eval_list, early_stopping_best, early_stopping_delta, patience_count, use_metric=True
        )

        assert best == 0.7
        assert patience == 0

    def test_update_early_stopping_no_improvement_metric(self):
        """Test early stopping update without improvement (metric)."""
        eval_list = [0.5, 0.6, 0.59]
        early_stopping_best = 0.6
        early_stopping_delta = 0.01
        patience_count = 0

        best, patience = _update_early_stopping(
            eval_list, early_stopping_best, early_stopping_delta, patience_count, use_metric=True
        )

        assert best == 0.6
        assert patience == 1
