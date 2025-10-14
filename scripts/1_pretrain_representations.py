"""Train the rnaseq preprocessor and rpz to be used in the prediction pipeline."""

# %%
from pathlib import Path

from loguru import logger

from configs.config_rpz_model import RPZ_MODEL
from leap.data.preclinical_dataset import PreclinicalDataset
from leap.data.preprocessor import OmicsPreprocessor
from leap.utils.config_utils import instantiate
from leap.utils.io import save_pickle


MODELS_PATH = Path(__file__).parent.parent / "models"


# %%
def train_preprocessor_and_rpz(
    studies: list[str],
    rpz_model_name: str,
    list_of_genes: str = "most_variant_genes",
    normalization: str = "tpm",
    random_seed: int = 0,
) -> None:
    """Train the rnaseq preprocessor and rpz."""
    # Define output file names
    output_path_preprocessor = (
        MODELS_PATH
        / "preprocessors"
        / (f"log_mean_std_{'_'.join(studies)}_{list_of_genes}_{normalization}_seed_{random_seed}.pkl")
    )
    output_path_rpz = (
        MODELS_PATH
        / "rpz"
        / (f"{rpz_model_name}_{'_'.join(studies)}_{list_of_genes}_{normalization}_seed_{random_seed}.pkl")
    )

    # Load the data
    gene_list = Path(__file__).parent.parent / "data" / f"list_of_{list_of_genes}.csv"
    X = PreclinicalDataset(label=None, normalization=normalization, use_gene_list=gene_list).df_rnaseq

    preprocessor = OmicsPreprocessor(
        scaling_method="mean_std",
        max_genes=-1,
        log_scaling=("combat" not in normalization),  # log-scaling is done before combat
    )
    preprocessor.fit(X=X)

    # Save the trained preprocessor
    save_pickle(preprocessor, output_path_preprocessor)

    # Transform the data using the trained preprocessor
    df_rnaseq_transformed = preprocessor.transform(X=X)

    # Define the rpz model config
    config_rpz_model = RPZ_MODEL[rpz_model_name]
    config_rpz_model.random_state = random_seed

    # Fit the rpz model
    logger.info("Fitting RPZ...")
    rpz_full = instantiate(config_rpz_model)
    rpz_full.fit(df_rnaseq_transformed)
    # Save the trained RPZ
    save_pickle(rpz_full, output_path_rpz)


# %%
if __name__ == "__main__":
    for random_seed in range(5):
        logger.info(f"Training preprocessor and mae with seed {random_seed}...")
        train_preprocessor_and_rpz(
            studies=["depmap"],
            rpz_model_name="mae",
            list_of_genes="most_variant_genes",
            normalization="tpm",
            random_seed=random_seed,
        )

# %%
