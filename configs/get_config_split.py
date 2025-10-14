"""Config for the training/few-shot/test splits to be used in the trainer."""

from typing import Literal

from ml_collections import config_dict

from leap.data.splits import cv_split_ids


def get_config_split(
    test_set_type: Literal["sample", "perturbation", "tissue", "transfer_learning"], training_split_count: int = -1
) -> config_dict.ConfigDict:
    """Create configuration for the training/few-shot/test splits.

    Parameters
    ----------
    test_set_type : Literal["sample", "perturbation", "tissue", "transfer_learning"]
        Type of test set to use. Possible values include:
            - "sample" for evaluation in unseen sample (cell lines or patients),
            - "perturbation" for evaluation in unseen perturbations,
            - "tissue" for evaluation in unseen tissues,
            - "transfer_learning" for evaluation in target data (unseen tissue, pdx or patient).
    training_split_count : int, optional
        Number of samples in the training set. Only used if test_set_type is "transfer_learning".

    Raises
    ------
    NotImplementedError
        If the test_set_type is not supported.

    Returns
    -------
    config : config_dict.ConfigDict
        Configuration for the data split.
    """
    if test_set_type == "sample":
        return _get_config_split_unseen_sample()
    if test_set_type == "perturbation":
        return _get_config_split_unseen_perturbation()
    if test_set_type == "tissue":
        return _get_config_split_unseen_tissue()
    if test_set_type == "transfer_learning":
        return _get_config_split_transfer_learning(training_split_count)
    raise NotImplementedError(f"Test set type {test_set_type} is not supported.")


def _get_config_split_unseen_sample() -> config_dict.ConfigDict:
    """Create configuration for test in unseen cell lines."""
    return _get_config_cv_split_ids(
        group_variable="sample",
        subgroup_variable=None,
        stratify_variable=None,
        test_split_ratio=0.2,
        training_split_count=None,
        n_splits=10,
    )


def _get_config_split_unseen_perturbation() -> config_dict.ConfigDict:
    """Create configuration for test in unseen perturbations."""
    return _get_config_cv_split_ids(
        group_variable="perturbation",
        subgroup_variable=None,
        stratify_variable=None,
        test_split_ratio=0.2,
        training_split_count=None,
        n_splits=10,
    )


def _get_config_split_unseen_tissue() -> config_dict.ConfigDict:
    """Create configuration for test in unseen tissues."""
    return _get_config_cv_split_ids(
        group_variable="sample",
        subgroup_variable=None,
        stratify_variable=None,
        test_split_ratio=1.0,
        training_split_count=None,
        n_splits=1,
    )


def _get_config_split_transfer_learning(training_split_count: int = -1) -> config_dict.ConfigDict:
    """Create configuration for test in unseen disease models."""
    training_split_count = 10 if training_split_count < 0 else training_split_count
    return _get_config_cv_split_ids(
        group_variable=None,
        subgroup_variable="perturbation",
        stratify_variable=None,
        test_split_ratio=None,
        training_split_count=training_split_count,
        n_splits=100,
    )


def _get_config_cv_split_ids(
    group_variable: str | None,
    subgroup_variable: str | None,
    stratify_variable: str | None,
    test_split_ratio: float | None,
    training_split_count: int | None,
    n_splits: int,
) -> config_dict.ConfigDict:
    """Create general configuration for cross-validation splits.

    The cv_split_ids function used to generate the splits takes as input the stacked sample metadata where rows
    correspond to sample x perturbation pairs. The group_variable indicates if the split should be done by sample or by
    perturbation.

    Parameters
    ----------
    group_variable : str | None
        Column name in X_metadata to use for groups.
    subgroup_variable : str | None
        Column name in X_metadata to use for splitting.
    stratify_variable : str | None
        Column name in X_metadata to use for stratification.
    test_split_ratio : float | None
        Ratio of the test set size over the total number of sample x perturbation pairs. This should be between 0 and 1.
    training_split_count : int | None
        Number of samples in the test set. Only used if test_split_ratio is None.
    n_splits : int
        Number of splits generate.

    Returns
    -------
    config : config_dict.ConfigDict
        Configuration for the data split.
    """
    return config_dict.ConfigDict(
        {
            "_target_": cv_split_ids,
            "_partial_": True,
            "k_fold": False,  # never used for our training / test splits
            "group_variable": group_variable,
            "subgroup_variable": subgroup_variable,
            "stratify_variable": stratify_variable,
            "leave_one_group_out": False,  # never used
            "test_split_ratio": test_split_ratio,
            "training_split_count": training_split_count,
            "n_splits": n_splits,
            "n_min_loo": None,  # never used
            "list_test_groups": None,  # never used
            "random_state": 0,
        }
    )
