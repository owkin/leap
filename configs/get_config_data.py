"""Config for the data to be used in the prediction pipeline."""

from pathlib import Path

from ml_collections import config_dict

from leap.data.preclinical_dataset import PreclinicalDataset


REPO_PATH = Path(__file__).parent.parent


def get_config_data(
    source_domain_studies: str | list[str],
    source_domain_label: str,
    list_of_perturbations: str,
    filter_available_fingerprints: bool,
    normalization: str,
    list_of_genes: str,
    target_domain_studies: str | list[str] | None = None,
    target_domain_label: str | None = None,
) -> config_dict.ConfigDict:
    """Create data configuration for the perturbation model trainer.

    Parameters
    ----------
    source_domain_studies : str | list[str]
        The list of studies to use in the source domain data.
    source_domain_label : str
        The label to use in the source domain data. Possible values are "gene_dependency", "gene_effect", "AAC", "AUC",
        "pIC50", "min_delta_tumor_volume", "minus_min_delta_tumor_volume".
    list_of_perturbations : str
        The list of perturbations to use.
    filter_available_fingerprints : bool
        Whether to filter the drugs with available fingerprints.
    normalization : str
        The normalization to use for rnaseq data. Possible values are "tpm" only.
    list_of_genes : str
        The list of rnaseq genes to use.
    target_domain_studies : str | list[str] | None
        The list of studies to use in the target domain data.
    target_domain_label : str | None
        The label to use in the target domain data. Possible values are "gene_dependency", "gene_effect", "AAC", "AUC",
        "pIC50", "min_delta_tumor_volume", "minus_min_delta_tumor_volume". None if no label is used.

    Returns
    -------
    config : config_dict.ConfigDict
        The data configuration.
    """
    # Check arguments
    if isinstance(source_domain_studies, str):
        source_domain_studies = [source_domain_studies]
    if isinstance(target_domain_studies, str):
        target_domain_studies = [target_domain_studies]

    # Initialise the configuration
    config = config_dict.ConfigDict()

    # Define the source domain data configuration
    config.source_domain_data = get_config_preclinical_dataset(
        list_of_studies=source_domain_studies,
        normalization=normalization,
        list_of_genes=list_of_genes,
        label=source_domain_label,
        list_of_perturbations=list_of_perturbations,
        filter_available_fingerprints=filter_available_fingerprints,
    )

    # Define the target domain data configuration
    if target_domain_studies is not None:
        config.target_domain_data = get_config_preclinical_dataset(
            list_of_studies=target_domain_studies,
            normalization=normalization,
            list_of_genes=list_of_genes,
            label=target_domain_label,
            list_of_perturbations=list_of_perturbations,
            filter_available_fingerprints=filter_available_fingerprints,
        )
    else:
        config.target_domain_data = None

    return config


def get_config_preclinical_dataset(
    list_of_studies: list[str],
    normalization: str,
    list_of_genes: str,
    label: str | None,
    list_of_perturbations: str | None,
    filter_available_fingerprints: bool,
) -> config_dict.ConfigDict:
    """Create general preclinical data configuration.

    This configuration can be used to instantiate a PreclinicalDataset object. For parameters description, see the
    above docstring.
    """
    if len(list_of_studies) != 1 or list_of_studies[0] != "DepMap_23Q4":
        raise NotImplementedError("Only DepMap_23Q4 is supported for now.")

    # Define the minimal number of samples per label
    if label is None:
        min_n_label = 0
    elif label in {"gene_dependency", "gene_effect"}:
        min_n_label = 50
    elif label in {"AAC", "AUC", "IC50", "pIC50"}:
        if "PRISM_2020" in list_of_studies:
            # Use only 15 samples as PRISM is used as an external test dataset
            min_n_label = 15
        else:
            min_n_label = 75
    elif label in {"min_delta_tumor_volume", "minus_min_delta_tumor_volume"}:
        min_n_label = 15
    else:
        raise ValueError(f"{label} is not available.")

    # Initialise the configuration
    config = config_dict.ConfigDict(
        {
            "_target_": PreclinicalDataset,
            "label": label,
            "normalization": normalization,
            "scale_label": False,  # default value
            "min_n_label": min_n_label,
            "filter_available_fingerprints": filter_available_fingerprints,
        }
    )

    # Define the gene list for rnaseq data
    config.use_gene_list = _get_path_list_of_genes_rnaseq(list_of_genes)

    # Define the labels list
    if label is not None:
        if list_of_perturbations is None:
            raise ValueError("list_of_perturbations must be set. if label is not None.")
        config.use_label_list = _get_path_list_of_perturbations(list_of_perturbations)

    return config


def _get_path_list_of_genes_rnaseq(list_of_genes: str) -> Path | None:
    """Get the path to the list of genes to use for rnaseq data."""
    # Return no gene list
    possible_gene_lists = {"all", "most_variant_genes"}

    if list_of_genes not in possible_gene_lists:
        raise ValueError(
            f"{list_of_genes} does not exis. Please provide a valid `list_of_genes` in {possible_gene_lists}."
        )

    if list_of_genes == "all":
        return None

    # Return the selected gene list
    file_path = REPO_PATH / "data" / f"list_of_{list_of_genes}.csv"
    if file_path.exists():
        return file_path

    raise NotImplementedError(f"{list_of_genes} cannot be found. Please add the file {file_path} to data/.")


def _get_path_list_of_perturbations(list_of_perturbations: str) -> Path | None:
    """Get the path to the list of perturbations to use."""
    possible_perturbations = {
        "all",
        "perturbations_task_1",
        "perturbations_task_2",
        "perturbations_task_3",
        "perturbations_task_4",
        "perturbations_task_5",
    }
    if list_of_perturbations not in possible_perturbations:
        raise ValueError(
            f"{list_of_perturbations} does not exis. Please provide a valid "
            f"`list_of_perturbations` in {possible_perturbations}."
        )

    if list_of_perturbations == "all":
        return None

    file_path = REPO_PATH / "data" / f"list_of_{list_of_perturbations}.csv"
    if file_path.exists():
        return file_path

    raise NotImplementedError(f"{list_of_perturbations} cannot be found. Please add the file {file_path} to data/.")
