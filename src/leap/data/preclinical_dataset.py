"""A standardised class for preclinical datasets."""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from leap.data.load_depmap import load_essentiality, load_expression, load_metadata
from leap.data.load_gene_sets import load_fingerprints


class PreclinicalDataset:
    """A preclinical dataset.

    A preclinical dataset including RNAseq and gene essentiality data from DepMap. Can be extended to other datasets,
    modalities and labels.

    Parameters
    ----------
    label : str | None, optional
        The label to load. Possible values are "gene_dependency". Default is "gene_dependency".
    scale_label : bool, optional
        Whether to scale a continuous label, by default False.
    min_n_label : int, optional
        The minimum number of non-missing labels to keep, by default 50.
    use_label_list : str | None, optional
        Path to the list of labels to use, by default None.
    use_gene_list_rnaseq : str | None, optional
        Path to the list of genes to use for RNAseq, by default None (using all available genes).
    filter_available_fingerprints : bool, optional
        Whether to keep only the labels with available fingerprints, by default False.
    tissues_to_keep : str | list[str], optional
        The tissues to keep, by default "all".
    tissues_to_exclude : str | list[str] | None, optional
        The tissues to exclude, by default None.

    Attributes
    ----------
    df_rnaseq : pd.DataFrame
        RNAseq expression data for cell lines.
    df_sample_metadata : pd.DataFrame
        Metadata for cell lines.
    df_labels : pd.DataFrame
        Gene essentiality labels (samples x genes).
    df_fingerprints : pd.DataFrame | None
        Fingerprints for genes (if available).
    df_labels_stacked : pd.DataFrame
        Stacked labels (sample x perturbation pairs).
    df_sample_metadata_stacked : pd.DataFrame
        Sample metadata aligned with stacked labels.
    """

    def __init__(
        self,
        label: str | None = "gene_dependency",
        scale_label: bool = False,
        min_n_label: int = 50,
        use_label_list: str | None = None,
        use_gene_list_rnaseq: str | None = None,
        filter_available_fingerprints: bool = False,
        tissues_to_keep: str | list[str] = "all",
        tissues_to_exclude: str | list[str] | None = None,
    ) -> None:
        self.label = label
        self.scale_label = scale_label
        self.min_n_label = min_n_label
        self.use_label_list = use_label_list
        self.use_gene_list_rnaseq = use_gene_list_rnaseq
        self.filter_available_fingerprints = filter_available_fingerprints
        self.tissues_to_keep = tissues_to_keep
        self.tissues_to_exclude = tissues_to_exclude

        # Define attributes
        self.df_labels: pd.DataFrame
        self.df_labels_stacked: pd.DataFrame
        self.df_sample_metadata: pd.DataFrame
        self.df_sample_metadata_stacked: pd.DataFrame
        self.df_fingerprints: pd.DataFrame | None
        self.df_rnaseq: pd.DataFrame

        # Load metadata
        self._load_sample_metadata()

        # Load labels if specified
        if self.label is not None:
            self._load_labels()
            self._load_fingerprints()
        else:
            self.df_labels = pd.DataFrame(index=self.df_sample_metadata.index)
            self.df_fingerprints = None

        # Load RNAseq
        self._load_rnaseq()
        if self.label is not None:
            logger.info(f"Loaded {self.df_labels.shape[1]} perturbations in {self.df_labels.shape[0]} samples.")
        if self.df_fingerprints is not None:
            logger.info(f"Loaded fingerprints for {self.df_fingerprints.shape[0]} perturbations.")

        # Filter samples based on tissue
        self._filter_tissues()
        # Align sample info with labels and rnaseq
        self.align_sample_data()
        # Sort rows and columns to ensure consistency
        self._sort_rows_and_columns()
        # Use float64 for rnaseq data
        self._format_dataframe()
        # Stack labels and align sample metadata
        self.stack_dataframes()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object.

        This method is called when unpickling an object, and is customized to handle
        the renaming of attributes.
        """
        # Handle None dataframes
        df_keys = [
            "df_labels",
            "df_labels_stacked",
            "df_sample_metadata",
            "df_sample_metadata_stacked",
            "df_fingerprints",
            "df_rnaseq",
        ]
        for key in df_keys:
            if key in state and state[key] is None:
                del state[key]
        self.__dict__.update(state)

    def stack_dataframes(self) -> None:
        """Stack labels and align sample metadata."""
        # Stack labels such that rows correspond to sample x perturbation pairs
        if self.label is not None:
            stacked_labels = self.df_labels.stack()
            stacked_labels.index.names = ["sample", "perturbation"]
            self.df_labels_stacked = pd.DataFrame(stacked_labels, columns=["label"])

            # Align a stacked sample metadata dataframe
            self.df_sample_metadata_stacked = self.df_labels_stacked.merge(
                self.df_sample_metadata, left_on="sample", right_index=True
            )[self.df_sample_metadata.columns]
            self.df_sample_metadata_stacked = pd.merge(
                self.df_sample_metadata_stacked, self.df_labels_stacked, on=["sample", "perturbation"], how="inner"
            )

            # Create sample and perturbation columns to make them more accessible
            self.df_sample_metadata_stacked["sample"] = self.df_sample_metadata_stacked.index.get_level_values("sample")
            self.df_sample_metadata_stacked["perturbation"] = self.df_sample_metadata_stacked.index.get_level_values(
                "perturbation"
            )
            self.df_sample_metadata_stacked["perturbation_label"] = (
                self.df_sample_metadata_stacked["perturbation"]
                + "_"
                + self.df_sample_metadata_stacked["label"].astype(str)
            )

    def _load_sample_metadata(self) -> None:
        """Load metadata on the samples from DepMap."""
        df_sample_metadata = load_metadata()
        logger.info(f"Loaded metadata for {df_sample_metadata.shape[0]} cell lines.")

        # Rename tissue column to match expected format
        if "OncotreeLineage" in df_sample_metadata.columns:
            df_sample_metadata = df_sample_metadata.rename(columns={"OncotreeLineage": "tissue"})

        self.df_sample_metadata = df_sample_metadata

    def _load_rnaseq(self) -> None:
        """Load RNAseq data from DepMap."""
        df_rnaseq = load_expression()
        logger.info(f"Loaded expression of {df_rnaseq.shape[1]} genes in {df_rnaseq.shape[0]} cell lines.")

        # Keep only the listed genes if specified
        self._use_gene_list_rnaseq(df_rnaseq)
        # Add rnaseq suffix
        self.df_rnaseq = df_rnaseq.add_suffix("_rnaseq")

    def _load_labels(self) -> None:
        """Load gene essentiality labels from DepMap."""
        if self.label == "gene_dependency":
            df_labels = load_essentiality()
        else:
            raise ValueError(f"Invalid label: {self.label}. Only 'gene_dependency' is supported for DepMap.")

        # Standardise perturbation names
        df_labels.columns = pd.Index([name.lower() for name in df_labels.columns])

        # Remove any infinite values
        df_labels.replace([-np.inf, np.inf], np.nan, inplace=True)

        if self.scale_label:
            scaler = StandardScaler()
            df_labels = pd.DataFrame(scaler.fit_transform(df_labels), columns=df_labels.columns, index=df_labels.index)

        # Keep only the listed labels
        self._use_label_list(df_labels)
        # Keep only samples with at least one label
        df_labels.dropna(how="all", inplace=True)
        self.df_labels = df_labels

    def _load_fingerprints(self) -> None:
        """Load fingerprints for genes from DepMap."""
        df_fingerprints = load_fingerprints()
        # Standardise perturbation names to match labels
        df_fingerprints.index = pd.Index([name.lower() for name in df_fingerprints.index])
        # Keep only fingerprints for genes in df_labels
        common_genes = list(set(df_fingerprints.index).intersection(set(self.df_labels.columns)))
        self.df_fingerprints = df_fingerprints.loc[common_genes]

        # Filter labels to only keep those with available fingerprints if requested
        if self.filter_available_fingerprints:
            self.df_labels = self.df_labels.loc[:, common_genes]
            logger.info(f"Filtered labels to keep only {len(common_genes)} genes with available fingerprints.")

        logger.info(f"Loaded fingerprints for {self.df_fingerprints.shape[0]} genes.")

    def _use_label_list(self, df_labels: pd.DataFrame) -> None:
        """Only keep the listed labels."""
        if self.use_label_list is not None:
            label_list = list(pd.read_csv(self.use_label_list, header=None).iloc[:, 0])
            label_list = [name.lower() for name in label_list]
            df_labels = df_labels.loc[:, df_labels.columns.isin(label_list)]
        self.df_labels = df_labels

    def _use_gene_list_rnaseq(self, df_rnaseq: pd.DataFrame) -> None:
        """Only keep the listed RNASeq genes."""
        if self.use_gene_list_rnaseq is not None:
            # Load the gene list from the specified file
            gene_list_rnaseq = (
                pd.read_csv(self.use_gene_list_rnaseq, header=None).iloc[:, 0].str.removesuffix("_rnaseq")
            )
            df_rnaseq = df_rnaseq.loc[:, df_rnaseq.columns.isin(gene_list_rnaseq)]

    def _get_common_samples(self, *dfs: pd.DataFrame) -> list[str]:
        """Get the common samples between dataframes."""
        common_samples = set(dfs[0].index)
        for df in dfs[1:]:
            if df is not None:
                common_samples.intersection_update(df.index)
        return sorted(common_samples)

    def align_sample_data(self) -> None:
        """Align sample metadata with labels and rnaseq data.

        This function ensures that df_rnaseq, df_labels, and df_sample_metadata have the same index.
        We consider two strategies:
        - If self.min_n_label > 0, we only keep samples with at least min_n_label
          labels. This is useful for training prediction models.
        - If self.min_n_label == 0, we keep all samples with rnaseq data. This is useful
          for training RPZ models.
        """
        if self.min_n_label == 0:
            self.df_labels = self.df_labels.reindex(self.df_rnaseq.index)

        common_samples = self._get_common_samples(
            self.df_sample_metadata,
            self.df_labels,
            self.df_rnaseq,
        )

        logger.info(
            f"{len(self.df_rnaseq) - len(common_samples)} rnaseq samples are"
            " dropped when aligning with labels and metadata."
        )
        logger.info(
            f"{len(self.df_labels) - len(common_samples)} labelled samples are dropped when aligning with metadata."
        )
        logger.info(
            f"{len(self.df_sample_metadata) - len(common_samples)} metadata"
            " samples are dropped when aligning with labels."
        )

        self.df_sample_metadata = self.df_sample_metadata.loc[common_samples]
        self.df_labels = self.df_labels.loc[common_samples]
        self.df_rnaseq = self.df_rnaseq.loc[common_samples]

        if self.min_n_label > 0:
            # Keep only labels with at least min_n_label non-missing values
            self.df_labels.dropna(axis=1, thresh=self.min_n_label, inplace=True)
            # Warning if df_labels is empty
            if len(self.df_labels.columns) == 0:
                logger.warning(
                    f"df_labels is empty after dropping labels with less than {self.min_n_label} non-missing values."
                )

    def _filter_tissues(self) -> None:
        """Filter samples based on tissues_to_keep and tissues_to_exclude."""

        def rename_for_code(x: str) -> str:
            """Normalize tissue names."""
            return str(x).lower().replace(" ", "_").replace("-", "_")

        # Keep only samples with a tissue in tissues_to_keep
        if self.tissues_to_keep != "all":
            self.df_sample_metadata = self.df_sample_metadata.dropna(subset="tissue")
            self.df_sample_metadata = self.df_sample_metadata.loc[
                self.df_sample_metadata["tissue"]
                .apply(rename_for_code)
                .isin(pd.Series(self.tissues_to_keep).apply(rename_for_code))
            ]

        # Remove samples with a tissue in tissues_to_exclude
        if self.tissues_to_exclude is not None:
            self.df_sample_metadata = self.df_sample_metadata.dropna(subset="tissue")
            self.df_sample_metadata = self.df_sample_metadata.loc[
                ~self.df_sample_metadata["tissue"]
                .apply(rename_for_code)
                .isin(pd.Series(self.tissues_to_exclude).apply(rename_for_code))
            ]

    def _sort_rows_and_columns(self) -> None:
        """Sort rows and columns for consistency."""
        self.df_sample_metadata = self.df_sample_metadata.sort_index(axis=0)
        self.df_rnaseq = self.df_rnaseq.sort_index(axis=0).sort_index(axis=1)
        if self.label is not None:
            self.df_labels = self.df_labels.sort_index(axis=0).sort_index(axis=1)
            if self.df_fingerprints is not None:
                self.df_fingerprints = self.df_fingerprints.sort_index(axis=0)

    def _format_dataframe(self) -> None:
        """Format dataframes with proper data types."""
        self.df_rnaseq = self.df_rnaseq.astype("float64")

    def keep_perturbations(self, perturbation_names: list) -> None:
        """Keep only the perturbations in the list in all dataframes.

        Parameters
        ----------
        perturbation_names : list
            The list of perturbations to keep.
        """
        self.df_labels = self.df_labels[perturbation_names]
        if self.df_fingerprints is not None:
            self.df_fingerprints = self.df_fingerprints.loc[self.df_fingerprints.index.intersection(perturbation_names)]
            if self.df_fingerprints.empty:
                self.df_fingerprints = None
        self._sort_rows_and_columns()
        self.stack_dataframes()

    def merge(self, data: "PreclinicalDataset") -> None:
        """Merge two PreclinicalDataset objects.

        Parameters
        ----------
        data : PreclinicalDataset
            The dataset to merge.
        """
        # Concatenate labels and sample metadata
        self.df_labels = pd.concat([self.df_labels, data.df_labels], axis=0)
        self.df_sample_metadata = pd.concat([self.df_sample_metadata, data.df_sample_metadata], axis=0)

        # Update df_labels_stacked and df_sample_metadata_stacked
        self._sort_rows_and_columns()
        self.stack_dataframes()

        # Concatenate molecular data
        if hasattr(data, "df_rnaseq"):
            # Concatenate molecular data
            common_columns = list(set(self.df_rnaseq.columns) & set(data.df_rnaseq.columns))
            # Keeps only common columns
            self.df_rnaseq = self.df_rnaseq[common_columns]
            data.df_rnaseq = data.df_rnaseq[common_columns]
            self.df_rnaseq = pd.concat([self.df_rnaseq, data.df_rnaseq], axis=0)
