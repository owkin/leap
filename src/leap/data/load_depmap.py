"""Loading functions for depmap data."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"


def load_expression() -> pd.DataFrame:
    """Load DepMap RNASeq data with tpm normalization.

    It is important to note that RNAseq tpm data from DepMap is already log-scaled with log2(X+1), therefore here we
    convert the tpm data with exp2(x) - 1 so that we can access the TPM values.
    """
    path_processed = DATA_PATH / "processed" / "tpm_rnaseq_processed.parquet"
    if path_processed.exists():
        depmap_expr = pd.read_parquet(path_processed)
    else:
        logger.info("Preprocessing DepMap RNASeq data with tpm normalization...")
        path = DATA_PATH / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
        depmap_expr = (
            pd.read_csv(path, index_col=0)
            .query("IsDefaultEntryForModel == 'Yes'")
            .drop(columns=["SequencingID", "IsDefaultEntryForModel", "ModelConditionID", "IsDefaultEntryForMC"])
            .rename(columns={"ModelID": "DepMap_ID"})
            .set_index("DepMap_ID")
            .apply(lambda x: np.exp2(x) - 1)
            .astype("float32")
        )
        # Clean column names to have gene symbol only
        depmap_expr.rename(columns=lambda x: re.sub(r"[\(\[ ].*?[\)\]]", "", x), inplace=True)
        depmap_expr.to_parquet(path_processed, engine="pyarrow", compression="brotli")
    return depmap_expr


def load_essentiality() -> pd.DataFrame:
    """Load DeepDEP essentiality scores."""
    labels_path_processed = DATA_PATH / "processed" / "dependencies_processed.parquet"
    labels_path_raw = DATA_PATH / "CRISPRGeneDependency.csv"

    if labels_path_processed.exists():
        gene_dependencies = pd.read_parquet(labels_path_processed)
    else:
        logger.info("Preprocessing DepMap CRISPR data...")
        gene_dependencies = pd.read_csv(labels_path_raw, index_col=0).rename(
            columns=lambda x: re.sub(r"[\(\[ ].*?[\)\]]", "", x)
        )
        gene_dependencies.index.names = ["DepMap_ID"]
        gene_dependencies.to_parquet(labels_path_processed, engine="pyarrow", compression="brotli")
    return gene_dependencies


def load_metadata() -> pd.DataFrame:
    """Load metadata for all of DepMap's cancer cell lines."""
    df_sample_info = pd.read_csv(DATA_PATH / "Model.csv", index_col=0)
    df_sample_info.index.names = ["DepMap_ID"]
    return df_sample_info
