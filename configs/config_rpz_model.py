"""Define configs for rpz models to use in the pipeline."""

from ml_collections import config_dict

from leap.representation_models import PCA, MaskedAutoencoder
from leap.utils import get_device


# Detect device (CPU, CUDA, or MPS)
DEVICE = get_device()

RPZ_MODEL: dict[str, config_dict.ConfigDict] = {
    "pca": config_dict.ConfigDict(
        {
            "_target_": PCA,
            "repr_dim": 256,
            "random_state": 0,
        }
    ),
    "mae": config_dict.ConfigDict(
        {
            "_target_": MaskedAutoencoder,
            "repr_dim": 256,
            "corruption_method": "vime",
            "corruption_proba": 0.3,
            "hidden_n_units_first": 512,
            "hidden_n_layers": 0,
            "early_stopping_use": True,
            "early_stopping_patience": 20,
            "early_stopping_delta": 1e-5,
            "max_num_epochs": 1000,
            "batch_size": 1024,
            "learning_rate": 1e-4,
            "retrain": True,
            "data_augmentation": False,
            "da_noise_std": 0.0,
            "dropout": 0.0,
            "device": DEVICE,
            "random_state": 0,
        }
    ),
    "ae": config_dict.ConfigDict(
        {
            "_target_": MaskedAutoencoder,
            "repr_dim": 256,
            "corruption_method": "classic",
            "corruption_proba": 0.0,
            "hidden_n_units_first": 512,
            "hidden_n_layers": 0,
            "early_stopping_use": True,
            "early_stopping_patience": 100,
            "early_stopping_delta": 1e-7,
            "max_num_epochs": 10000,
            "batch_size": 1024,
            "learning_rate": 1e-4,
            "retrain": True,
            "data_augmentation": False,
            "da_noise_std": 0.01,
            "dropout": 0.2,
            "device": DEVICE,
            "random_state": 0,
        }
    ),
}
