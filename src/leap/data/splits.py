"""Implementation of the splits for the perturbation model."""

import random
from collections.abc import Generator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


def _validate_cv_split_params(  # noqa: PLR0912
    k_fold: bool,
    group_variable: str | None,
    subgroup_variable: str | None,
    stratify_variable: str | None,
    leave_one_group_out: bool,
    test_split_ratio: float | None,
    training_split_count: int | None,
    n_splits: int | None,
) -> None:
    """Validate cross-validation split parameters.

    Parameters
    ----------
    k_fold : bool
        Whether to use K-fold splits.
    group_variable : str | None
        Column name for grouping.
    subgroup_variable : str | None
        Column name for subgrouping.
    stratify_variable : str | None
        Column name for stratification.
    leave_one_group_out : bool
        Whether to use leave-one-group-out.
    test_split_ratio : float | None
        Ratio of test set size.
    training_split_count : int | None
        Number of training samples.
    n_splits : int | None
        Number of splits to generate.

    Raises
    ------
    ValueError
        If parameter combinations are invalid.
    NotImplementedError
        If feature combinations are not yet supported.
    """
    # Validate leave_one_group_out compatibility
    if leave_one_group_out:
        if group_variable is None:
            raise ValueError("Cannot leave one group out without a grouping variable.")
        if stratify_variable is not None:
            raise NotImplementedError("Cannot stratify and leave one group out at the same time.")
        if subgroup_variable is not None:
            raise NotImplementedError("Cannot split by subgroup and leave one group out at the same time.")
        if n_splits is not None:
            raise ValueError("Cannot fix the number of splits with leave one group out.")
        if test_split_ratio is not None:
            raise ValueError("Cannot fix the test split ratio with leave one group out.")
        if training_split_count is not None:
            raise ValueError("Cannot fix the number of training samples with leave one group out.")
        if k_fold:
            raise ValueError("Cannot do K fold and leave one group out at the same time.")

    # Validate k_fold compatibility
    if k_fold:
        if test_split_ratio is not None:
            raise ValueError("Cannot fix the test split ratio with K folds.")

    # Validate test_split_ratio
    if test_split_ratio is not None:
        if test_split_ratio <= 0:
            raise ValueError("Test split ratio must be > 0.")
        if test_split_ratio >= 1:
            raise ValueError("Test split ratio must be < 1.")

    # Validate training_split_count compatibility
    if training_split_count is not None:
        if test_split_ratio is not None:
            raise ValueError("Either test_split_ratio or training_split_count should be provided, not both.")
        if k_fold:
            raise ValueError("Cannot use k_fold if training_split_count is provided.")
        if leave_one_group_out:
            raise ValueError("Cannot use leave_one_group_out if training_split_count is provided.")

    # Validate subgroup_variable compatibility
    if subgroup_variable is not None:
        if training_split_count is None:
            raise ValueError("training_split_count must be provided when using subgroup_variable.")

    # Validate group + stratify combination
    if test_split_ratio is not None and group_variable is not None and stratify_variable is not None:
        raise NotImplementedError("Cannot stratify and group at the same time.")


def cv_split_ids(
    X_metadata: pd.DataFrame,
    k_fold: bool = False,
    group_variable: str | None = None,
    subgroup_variable: str | None = None,
    stratify_variable: str | None = None,
    leave_one_group_out: bool = False,
    test_split_ratio: float = 0.2,
    training_split_count: int | None = None,
    n_splits: int = 10,
    n_min_loo: int = 15,
    list_test_groups: Path | None = None,
    random_state: int = 0,
) -> dict[str, dict[str, list]]:
    """Training and test indices of the dataset for cross-validation.

    This function supports shuffle, group shuffle, stratified shuffle splits, K fold or group K fold. It can also be
    used for leave-one-group-out cross validation.

    Parameters
    ----------
    X_metadata : pd.DataFrame
        Input data.
    k_fold : bool
        Whether to split the data into K folds, by default False. If False, the data is split into non-overlapping
        training and test sets.
    group_variable : str | None
        Column name in X_metadata to use for groups, by default None.
    subgroup_variable : str | None
        Column name in X_metadata to split data into subgroups for selecting a fixed number of training samples
        (training_split_count) per subgroup, by default None.
    stratify_variable : str | None
        Column name in X_metadata to use to stratify, by default None.
    leave_one_group_out : bool
        Whether to perform leave-one-group-out, by default False. If True, group_variable must be provided.
    test_split_ratio : float
        Ratio of the test set size over the total sample size in df_all, by default 0.2. This should be between 0 and 1
        or should be set to None for leave-one-group-out.
    training_split_count : int | None
        Number of samples in the test set. Only used if test_split_ratio is None.
    n_splits : int
        Number of splits generate, by default 10. This should be set to None for leave-one-group-out. This corresponds
        to the number of folds if k_fold is True.
    n_min_loo : int
        Minimum number of samples to keep a test set in leave-one-group-out, by default 15. We only keep splits that
        have at least n_min_loo samples for all perturbations.
    list_test_groups : Path | None
        Path to a csv file listing the groups to use as test sets. Only used if leave_one_group_out is True.
    random_state : int
        Random seed used by the split generator, by default 0.

    Raises
    ------
    ValueError
        If leave_one_group_out is True and group_variable is None.
        If leave_one_group_out is True and stratify_variable is not None.
        If leave_one_group_out is True and n_splits is not None.
        If leave_one_group_out is True and test_split_ratio is not None.
        If test_split_ratio is 0 and n_splits is not 1.
        If test_split_ratio is 1 and n_splits is not 1.

    Returns
    -------
    dict[str, dict[str, list]]
        Dictionary with training and test set ids.
    """
    # Initialise the dictionary to store the split ids
    split_ids: dict[str, dict[str, list]] = {}

    # Store the split ids
    if test_split_ratio == 0:
        if n_splits != 1:
            raise ValueError("Cannot have more than one split with test_split_ratio = 0.")

        # Use all samples for training
        split_ids["split_0"] = {"training_ids": X_metadata.index.to_list(), "test_ids": []}
    elif test_split_ratio == 1:
        if n_splits != 1:
            raise ValueError("Cannot have more than one split with test_split_ratio = 1.")

        # Use all samples for test
        split_ids["split_0"] = {"training_ids": [], "test_ids": X_metadata.index.to_list()}
    else:
        # Create the generator
        split_generator = cv_split_generator(
            X_metadata=X_metadata,
            k_fold=k_fold,
            group_variable=group_variable,
            subgroup_variable=subgroup_variable,
            stratify_variable=stratify_variable,
            leave_one_group_out=leave_one_group_out,
            test_split_ratio=test_split_ratio,
            training_split_count=training_split_count,
            n_splits=n_splits,
            n_min_loo=n_min_loo,
            list_test_groups=list_test_groups,
            return_group=True,
            random_state=random_state,
        )

        # Extract the training and test ids from the generator
        if leave_one_group_out:
            for training_ids, test_ids, loo_group in split_generator:
                split_ids[f"split_{loo_group}"] = {
                    "training_ids": X_metadata.index[training_ids].to_list(),
                    "test_ids": X_metadata.index[test_ids].to_list(),
                    # Storing information on the group used as test set
                    "ood_test": loo_group,
                }
        else:
            for split in range(n_splits):
                training_ids, test_ids = next(split_generator)
                split_ids[f"split_{split}"] = {
                    "training_ids": X_metadata.index[training_ids].to_list(),
                    "test_ids": X_metadata.index[test_ids].to_list(),
                }

    return split_ids


def cv_split_generator(
    X_metadata: pd.DataFrame,
    k_fold: bool = False,
    group_variable: str | None = None,
    subgroup_variable: str | None = None,
    stratify_variable: str | None = None,
    leave_one_group_out: bool = False,
    test_split_ratio: float | None = 0.2,
    training_split_count: int | None = None,
    n_splits: int | None = 10,
    n_min_loo: int = 15,
    list_test_groups: Path | None = None,
    return_group: bool = False,
    random_state: int = 0,
) -> Generator:
    """Create generator for cross validation.

    This function supports shuffle, group shuffle, stratified shuffle splits, K fold or group K fold. It can also be
    used for leave-one-group-out cross validation.

    Parameters
    ----------
    X_metadata : pd.DataFrame
        Input data.
    k_fold : bool
        Whether to split the data into K folds, by default False. If False, the data is split into non-overlapping
        training and test sets.
    group_variable : str | None
        Column name in X_metadata to use for groups, by default None.
    subgroup_variable : str | None
        Column name in X_metadata to split data into subgroups for selecting a fixed number of training samples
        (training_split_count) per subgroup, by default None.
    stratify_variable : str | None
        Column name in X_metadata to use to stratify, by default None.
    leave_one_group_out : bool
        Whether to perform leave-one-group-out, by default False. If True, group_variable must be provided.
    test_split_ratio : float | None
        Ratio of the test set size over the total sample size in df_all, by default 0.2. This should be strictly between
        0 and 1 or should be set to None for leave-one-group-out.
    training_split_count : int | None
        Number of samples in the test set. Only used if test_split_ratio is None.
    n_splits : int | None
        Number of splits generate, by default 10. This should be set to None for leave-one-group-out. This corresponds
        to the number of folds if k_fold is True.
    n_min_loo : int
        Minimum number of samples to keep a test set in leave-one-group-out, by default 15. We only keep splits that
        have at least n_min_loo samples for all perturbations.
    list_test_groups : Path | None
        Path to a csv file listing the groups to use as test sets. Only used if leave_one_group_out is True.
    return_group : bool
        Whether to return the group value, by default False.
    random_state : int
        Random seed used by the split generator, by default 0.

    Returns
    -------
    Generator
        Generator for training and test set indexes.
    """
    # Validate all parameters
    _validate_cv_split_params(
        k_fold=k_fold,
        group_variable=group_variable,
        subgroup_variable=subgroup_variable,
        stratify_variable=stratify_variable,
        leave_one_group_out=leave_one_group_out,
        test_split_ratio=test_split_ratio,
        training_split_count=training_split_count,
        n_splits=n_splits,
    )

    # Generate training and test splits
    if test_split_ratio is not None:
        # Create data-specific split generator
        if group_variable is not None:
            # Use group shuffle split
            split_iterator = GroupShuffleSplit(n_splits=n_splits, test_size=test_split_ratio, random_state=random_state)
            split_generator = split_iterator.split(X=np.arange(len(X_metadata)), groups=X_metadata[group_variable])
        elif stratify_variable is not None:
            # Use stratified shuffle split
            split_iterator = StratifiedShuffleSplit(
                n_splits=n_splits, test_size=test_split_ratio, random_state=random_state
            )
            split_generator = split_iterator.split(X=np.arange(len(X_metadata)), y=X_metadata[stratify_variable])
        else:
            # Use shuffle split
            split_iterator = ShuffleSplit(n_splits=n_splits, test_size=test_split_ratio, random_state=random_state)
            split_generator = split_iterator.split(X=np.arange(len(X_metadata)))
    elif subgroup_variable is not None and training_split_count is not None and n_splits is not None:
        split_generator = _leave_n_samples_out_split_generator(
            X_metadata=X_metadata,
            subgroup_variable=subgroup_variable,
            training_split_count=training_split_count,
            n_splits=n_splits,
            stratify_variable=stratify_variable,
            random_state=random_state,
        )
    elif leave_one_group_out and group_variable is not None:
        # Use leave one group out
        split_generator = leave_one_group_out_split_generator(
            df_all=X_metadata,
            group_variable=group_variable,
            n_min_loo_variable="perturbation",
            n_min_loo=n_min_loo,
            list_test_groups=list_test_groups,
            return_group=return_group,
        )
    elif k_fold:
        if group_variable is not None:
            # Use group K fold
            group_k_fold_iterator = GroupKFold(n_splits=n_splits)
            split_generator = group_k_fold_iterator.split(
                X=np.arange(len(X_metadata)), groups=X_metadata[group_variable]
            )
        elif stratify_variable is not None:
            # Use stratified K fold
            stratified_k_fold_iterator = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_generator = stratified_k_fold_iterator.split(
                X=np.arange(len(X_metadata)), y=X_metadata[stratify_variable]
            )
        else:
            # Use K fold
            k_fold_iterator = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_generator = k_fold_iterator.split(X=np.arange(len(X_metadata)))

    return split_generator


def _leave_one_group_out_split_generator(
    df_all: pd.DataFrame,
    group_variable: str,
    n_min_loo: int = 15,
    n_min_loo_variable: str | None = None,
    list_test_groups: Path | None = None,
) -> Generator[tuple[list, list, str]]:
    """Leave-one-group-out split generator.

    This function generates training and test splits for leave-one-group-out cross-validation.

    Parameters
    ----------
    df_all : pd.DataFrame
        Input dataset to split in training/test sets.
    group_variable : str
        Column name to use for groups.
    n_min_loo : int
        Minimum number of samples to keep a test set in leave-one-group-out, by default 15.
    n_min_loo_variable : str | None
        Column name to use to stratify when checking the number of available samples, by default None.
    list_test_groups : Path | None
        Path to a csv file listing the groups to use as test sets.

    Yields
    ------
    Generator[tuple[list, list, str]]
        Training and test set indexes and the group value.

    Returns
    -------
    Generator[tuple[list, list, str]]
        Training and test set indexes and the group value.
    """
    # Define the list of groups that can be used as test sets
    unique_values_of_ood = df_all[group_variable].unique().tolist()
    if list_test_groups is not None:
        # Only the groups listed in the csv file can be used as test sets
        test_groups = pd.read_csv(list_test_groups, header=None)[0]
        unique_values_of_ood = list(set(unique_values_of_ood).intersection(test_groups))
    for value in unique_values_of_ood:
        # Check the number of available samples
        if n_min_loo_variable is not None:
            # Check the minimum number of samples by perturbation for this tissue
            # We want to make sure that we have at least n_min samples per perturbation
            # and per tissue. Otherwise, the tissue is not used as a test set.
            n_min = df_all[[group_variable, n_min_loo_variable]].reset_index(drop=True).value_counts()[value].min()
        else:
            # Check the number of available samples for this tissue
            n_min = df_all[group_variable].value_counts()[value]

        # Check if there are enough samples in the tissue
        if n_min >= n_min_loo:
            # Create the generator with training ids, test ids and the tissue name
            df_all = df_all.reset_index(drop=True)
            test = df_all[df_all[group_variable] == value]
            train = df_all[df_all[group_variable] != value]
            yield train.index.tolist(), test.index.tolist(), value


def _leave_on_group_out_split_generator_no_group(
    df_all: pd.DataFrame,
    group_variable: str,
    n_min_loo: int = 15,
    n_min_loo_variable: str | None = None,
    list_test_groups: Path | None = None,
) -> Generator[tuple[list, list]]:
    """Leave-one-group-out split generator without group output.

    This function generates training and test splits for leave-one-group-out cross-validation.

    Parameters
    ----------
    df_all : pd.DataFrame
        Input dataset to split in training/test sets.
    group_variable : str
        Column name to use for groups.
    n_min_loo : int
        Minimum number of samples to keep a test set in leave-one-group-out, by default 15.
    n_min_loo_variable : str | None
        Column name to use to stratify when checking the number of available samples, by default None.
    list_test_groups : Path | None
        Path to a csv file listing the groups to use as test sets.

    Yields
    ------
    Generator[tuple[list, list]]
        Training and test set indexes.

    Returns
    -------
    Generator[tuple[list, list]]
        Training and test set indexes.
    """
    generator = _leave_one_group_out_split_generator(
        df_all=df_all,
        group_variable=group_variable,
        n_min_loo=n_min_loo,
        n_min_loo_variable=n_min_loo_variable,
        list_test_groups=list_test_groups,
    )
    for training_ids, test_ids, _ in generator:
        yield training_ids, test_ids


def leave_one_group_out_split_generator(
    df_all: pd.DataFrame,
    group_variable: str,
    n_min_loo: int = 15,
    n_min_loo_variable: str | None = None,
    list_test_groups: Path | None = None,
    return_group: bool = False,
) -> Generator[tuple[list, list, str]] | Generator[tuple[list, list]]:
    """Leave-one-group-out split generator.

    This function generates training and test splits for leave-one-group-out cross
    validation.

    Parameters
    ----------
    df_all : pd.DataFrame
        Input dataset to split in training/test sets.
    group_variable : str
        Column name to use for groups.
    n_min_loo : int
        Minimum number of samples to keep a test set in leave-one-group-out, by default 15.
    n_min_loo_variable : str | None
        Column name to use to stratify when checking the number of available samples, by default None.
    list_test_groups : Path | None
        Path to a csv file listing the groups to use as test sets.
    return_group : bool
        Whether to return the group value, by default False.

    Returns
    -------
    Generator[tuple[list, list, str]] | Generator[tuple[list, list]]
        Training and test set indexes and the group value.
    """
    if return_group:
        return _leave_one_group_out_split_generator(
            df_all=df_all,
            group_variable=group_variable,
            n_min_loo=n_min_loo,
            n_min_loo_variable=n_min_loo_variable,
            list_test_groups=list_test_groups,
        )
    else:
        return _leave_on_group_out_split_generator_no_group(
            df_all=df_all,
            group_variable=group_variable,
            n_min_loo=n_min_loo,
            n_min_loo_variable=n_min_loo_variable,
            list_test_groups=list_test_groups,
        )


def _leave_n_samples_out_split_generator(
    X_metadata: pd.DataFrame,
    subgroup_variable: str,
    training_split_count: int,
    n_splits: int,
    stratify_variable: str | None = None,
    random_state: int = 0,
) -> Generator[tuple[list, list]]:
    """Leave n samples out split generator.

    Parameters
    ----------
    X_metadata : pd.DataFrame
        Input data.
    subgroup_variable : str
        Column name in X_metadata to split data into subgroups for selecting a fixed number of training samples
        (training_split_count) per subgroup.
    training_split_count : int
        Number of samples in the test set.
    n_splits : int
        Number of splits generate, by default 10. This should be set to None for leave-one-group-out. This corresponds
        to the number of folds if k_fold is True.
    stratify_variable : str | None
        Column name in X_metadata to use to stratify, by default None.
    random_state : int
        Random seed used by the split generator, by default 0.

    Yields
    ------
    Generator[tuple[list, list]]
        Training and test set indexes.
    """
    # Set the seed
    random.seed(random_state)

    # Iterate over the number of splits
    for split in range(n_splits):
        # Get training and test set ids per stratum
        split_per_stratum = (
            X_metadata.reset_index(drop=True)
            .groupby(subgroup_variable)
            .apply(
                partial(
                    _group_shuffle_split_by_count,
                    training_split_count=training_split_count,
                    stratify_variable=stratify_variable,
                    random_state=split,
                ),
                include_groups=False,
            )
        )

        # Combining the splits per stratum into a single split
        training_ids: list[Any] = []
        test_ids: list[Any] = []
        for perturb_dict in split_per_stratum:
            training_ids = training_ids + perturb_dict["training_ids"]
            test_ids = test_ids + perturb_dict["test_ids"]

        yield training_ids, test_ids


def _group_shuffle_split_by_count(
    df_by_stratum: pd.DataFrame,
    training_split_count: int,
    stratify_variable: str | None = None,
    random_state: int | None = None,
) -> dict[str, list]:
    """Select training samples by count with optional stratification.

    Parameters
    ----------
    df_by_stratum : pd.DataFrame
        Data for a single stratum.
    training_split_count : int
        Number of samples to select for training.
    stratify_variable : str | None
        Column name to use for stratification, by default None.
    random_state : int | None
        Random seed, by default None.

    Returns
    -------
    dict[str, list]
        Dictionary with 'training_ids' and 'test_ids' lists.
    """
    if stratify_variable is None:
        training_ids = random.sample(list(df_by_stratum.index), k=training_split_count)
    else:
        split_iterator = StratifiedShuffleSplit(n_splits=1, test_size=training_split_count, random_state=random_state)

        # Generate the stratified split
        train_idx, _ = next(split_iterator.split(df_by_stratum, df_by_stratum[stratify_variable]))
        training_ids = df_by_stratum.index[train_idx].tolist()

    test_ids = list(set(df_by_stratum.index).difference(training_ids))

    return {"training_ids": training_ids, "test_ids": test_ids}
