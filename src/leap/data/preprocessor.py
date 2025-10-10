"""Preprocessing steps and basic feature selection for Omics data."""

from typing import Self

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler


SCALERS: dict[str, MinMaxScaler | StandardScaler | FunctionTransformer] = {
    "min_max": MinMaxScaler(),
    "mean_std": StandardScaler(with_std=True),
    "mean": StandardScaler(with_std=False),
    "identity": FunctionTransformer(func=None),
}


class OmicsPreprocessor:
    """Preprocesses normalized Omics data.

    Transformations:
        1. Select genes in given gene list or protein-coding genes, if specified.
        2. Normalize data with provided scaler if pre_normalize == True.
        3. Compute ranks of genes according to gene_filtering method.
        4. Select top max_genes genes based on their ranks.
        3. If log_scaling == True, apply log(x+1).
        4. Data is centered with the chosen scaling method.

    Parameters
    ----------
    scaling_method : str
        Scaling method to apply after the log transformation (min_max, mean_std, mean), by default "min_max".
    max_genes : int
        Number of genes with highest variance to keep. Keep all genes if `max_genes <= 0`, by default -1.
    log_scaling : bool
        If True, apply a log transformation to the data (x: log(x+1)), by default True.
    pre_normalize : bool
        If True, normalizes the data before selecting genes with the gene_filtering method, by default False.
    gene_list_source : str | list[str] | None
        Path to CSV file containing a list of genes to be considered, or list of genes, by default None.

    Raises
    ------
    ValueError
        If the scaler method is unknown.
    """

    def __init__(
        self,
        scaling_method: str = "min_max",
        max_genes: int = -1,
        log_scaling: bool = True,
        pre_normalize: bool = False,
        gene_list_source: str | list[str] | None = None,
    ):
        self.scaling_method = scaling_method
        self.max_genes = max_genes
        self.gene_list: list[str] | None = None
        if gene_list_source:
            if isinstance(gene_list_source, str):
                # Path to file containing the list of genes
                self.gene_list = pd.read_csv(gene_list_source, header=None).iloc[:, 0].tolist()
            else:
                self.gene_list = gene_list_source

        if self.scaling_method not in SCALERS:
            raise ValueError(f"Scaling method must be {SCALERS.keys()}, got '{self.scaling_method}'")

        self.scaler = SCALERS[self.scaling_method]
        self.log_scaling = log_scaling
        self.columns_to_keep: list[str] = []
        self.pre_normalize = pre_normalize

    def fit(self, X: pd.DataFrame) -> Self:
        """Compute gene list and fit the scaler used for later transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Omics data (untransformed).

        Returns
        -------
        Self
            self
        """
        # If the processor has already been used in pretraining or DA for instance skip.
        if not self.columns_to_keep:
            # Do gene filtering on pre specific gene list.
            if self.gene_list is not None:
                X = X[X.columns.intersection(self.gene_list)]

            # Do specific filtering like variance or wasserstein.
            if self.max_genes > 0:
                gene_ranks = self.rank_genes(X)
                logger.info(
                    f"Selecting {self.max_genes} genes based on"
                    f" variance filtering" + (", pre-normalized" if self.pre_normalize else "") + "."
                )
                # Save columns to keep for future fit_transform.
                # The sort values is going to "revert" the second argsort in the
                # rank_genes function. You will have the index of the highest variant
                # columns: [index_of_most_variant_genes ,.._second_most_variant, etc]
                # Sorting the columns to ensure the order is consistent.
                self.columns_to_keep = sorted(gene_ranks.sort_values()[: self.max_genes].index.tolist())
            else:
                # No gene filtering so you take all columns of X.
                # Save columns to keep for future fit_transform.
                # Sorting the columns to ensure the order is consistent.
                self.columns_to_keep = sorted(X.columns.tolist())

        if not set(self.columns_to_keep) <= set(X.columns):
            logger.warning("X does not have all the columns to keep")
            self.columns_to_keep = X.columns.intersection(self.columns_to_keep).tolist()
        X = X[self.columns_to_keep]
        # log transform
        if self.log_scaling is True:
            X = X.apply(np.log1p)

        # train scaler
        self.scaler.fit(X)

        return self

    def rank_genes(self, X: pd.DataFrame) -> pd.Series:
        """Rank genes.

        Rank genes according to each method contained in self.gene_filtering, then take the minimum rank of each gene
        across methods.

        Parameters
        ----------
        X : pd.DataFrame
            Omics data (Filtered on gene list, possibly log-scaled).

        Returns
        -------
        pd.Series
            gene rank.
        """
        gene_ranks = {}
        variances = np.var(X, axis=0)
        # Ranks, highest variance first
        # sorts the elements of the "variances" array in ascending order and
        # returns the indices that would sort the array. Then redo an argsort
        # to have the indices of the largest variances ranked.
        ranks = (-variances).argsort().argsort()
        # [rank_of_gene1,rank_of_gene2]
        # This is needed to combine multiple filtering together.
        # Add ranks to results dict
        gene_ranks["variance"] = ranks
        # Perform union of methods : take min ranks across methods
        min_gene_ranks: pd.Series = pd.concat([ranks for _, ranks in gene_ranks.items()], axis=1).apply(min, axis=1)

        return min_gene_ranks

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform gene selection and data scaling with the chosen method.

        Parameters
        ----------
        X : pd.DataFrame
            Omics data (untransformed).

        Returns
        -------
        pd.DataFrame
            Transformed Omics data after gene selection and scaling.
        """
        X = X[self.columns_to_keep].copy()
        if self.log_scaling:
            X = X.apply(np.log1p)

        # Keep X a pd DataFrame after scaling, not a np ndarray
        X = pd.DataFrame(self.scaler.transform(X.astype(float)), columns=X.columns, index=X.index)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute gene list to be kept, fit scaler, and apply data transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Omics data (untransformed).

        Returns
        -------
        pd.DataFrame
            Transformed data after gene selection and scaling.
        """
        self.fit(X)
        return self.transform(X)
