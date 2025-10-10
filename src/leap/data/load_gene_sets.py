"""Loading functions for MSigDB gene set data."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger


DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"


def load_fingerprints() -> pd.DataFrame:
    """Load MSigDB gene set membership matrix as fingerprints.

    Returns a binary DataFrame where rows are genes, columns are gene set names,
    and values are 1 if the gene belongs to the gene set, 0 otherwise.

    The data is cached as a processed parquet file for faster loading.

    Returns
    -------
    pd.DataFrame
        Binary membership matrix with shape (n_genes, n_gene_sets).
        Index: gene symbols
        Columns: gene set names
        Values: 1 (gene in set) or 0 (gene not in set)
    """
    path_processed = DATA_PATH / "processed" / "gene_set_fingerprints_processed.parquet"

    if path_processed.exists():
        fingerprints = pd.read_parquet(path_processed)
    else:
        logger.info("Processing MSigDB gene sets to create fingerprint matrix...")
        path = DATA_PATH / "c2.cgp.v2025.1.Hs.json"

        # Load raw JSON data
        with open(path) as f:
            raw_data = json.load(f)

        logger.info(f"Loaded {len(raw_data)} gene sets")

        # Collect all unique genes
        all_genes: set[str] = set()
        for gene_set_data in raw_data.values():
            all_genes.update(gene_set_data["geneSymbols"])

        all_genes_list = sorted(all_genes)  # Sort for consistent ordering
        gene_set_names = list(raw_data.keys())

        logger.info(f"Creating membership matrix with {len(all_genes_list)} genes and {len(gene_set_names)} gene sets")

        # Create binary matrix efficiently
        data = {gene_set_name: [0] * len(all_genes_list) for gene_set_name in gene_set_names}
        gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes_list)}

        for gene_set_name, gene_set_data in raw_data.items():
            for gene in gene_set_data["geneSymbols"]:
                idx = gene_to_idx[gene]
                data[gene_set_name][idx] = 1

        # Create DataFrame
        fingerprints = pd.DataFrame(data, index=all_genes_list)
        fingerprints.index.name = "gene"

        # Save to parquet for faster loading next time
        path_processed.parent.mkdir(parents=True, exist_ok=True)
        fingerprints.to_parquet(path_processed, engine="pyarrow", compression="brotli")

        logger.info(f"Created and cached fingerprint matrix with shape {fingerprints.shape}")
        logger.info(f"Matrix density: {fingerprints.sum().sum() / (fingerprints.shape[0] * fingerprints.shape[1]):.2%}")

    return fingerprints
