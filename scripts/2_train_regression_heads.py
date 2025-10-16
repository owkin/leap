"""Launch jobs for a given model and task."""

# %%
import os
import sys


# CRITICAL: Set environment variables BEFORE any other imports
# These prevent threading conflicts between Ray and PyTorch on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Prevent Ray from overriding GPU env vars when num_gpus=0
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.config_perturbation_model import PRED_MODEL_NAME
from configs.get_config import get_config
from leap.data.preclinical_dataset import PreclinicalDataset, rename_for_code
from leap.trainer.perturbation_model_trainer import PerturbationModelTrainer, SplitPairIds
from leap.utils.config_utils import instantiate
from leap.utils.io import save_pickle


REPO_PATH = Path(__file__).parent.parent


# %%
def get_task_parser() -> argparse.Namespace:
    """Argument parser for the pred_task function."""
    # Simulate passing arguments if running the script as a notebook
    if "2_train_regression_heads.py" not in sys.argv[0]:
        sys.argv = ["2_train_regression_heads.py"]

    # Parse arguments
    parser = argparse.ArgumentParser(description="Perturbation response prediction.")
    parser.add_argument("--task_id", type=str, default="1")
    parser.add_argument("--model_id", type=str, choices=PRED_MODEL_NAME.keys(), default="mae_ps_enet")
    parser.add_argument("--rpz_random_state", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()
    return args


# %%
def get_output_path(task_id: str, model_id: str, rpz_random_state: int) -> str:
    """Define the path where results will be saved.

    The path is defined as output_path / experiment_name / date_time. The output_path is retrieved from the repo
    structure. The experiment_name contains the task_id, model_id and rpz_random_state.
    """
    # Define parent folder name where results will be saved
    experiment_name = f"task_{task_id}_model_{model_id}_seed_{rpz_random_state}"
    # Create a date-time stamp
    date_time = time.strftime("%Y%m%d_%H%M%S")
    return str(REPO_PATH / "results" / experiment_name / date_time)


def run_task(
    task_id: str,
    model_id: str,
    rpz_random_state: int,
    save_perf: bool = True,
    save_trainer: bool = True,
    output_path: str = "",
) -> PerturbationModelTrainer:
    """Run a perturbation response prediction task.

    This function defines the configuration based on all arguments and runs the trainer. A log file is created and
    completed during the run. The trainer is saved at the end of the run.

    Parameters
    ----------
    task_id : str
        The task id to use. Possible values are listed in the config_trainer file.
    model_id : str
        The model id to use. Possible values are listed in the config_perturbation_model file.
    rpz_random_state : int
        The random state to use.
    save_perf : bool
        Whether to save the performances, by default True.
    save_trainer : bool
        Whether to save the trainer, by default True.
    output_path : str, optional
        Path to the output folder where the results are saved. A default path is defined if output_path is "".

    Returns
    -------
    trainer : PerturbationModelTrainer
        The trainer including data, split ids, trained models and evaluation metrics.
    """
    # Define the config
    config = get_config(task_id=task_id, model_id=model_id, rpz_random_state=rpz_random_state)

    # Define the output path
    if output_path == "":
        output_path = get_output_path(task_id=task_id, model_id=model_id, rpz_random_state=rpz_random_state)

    trainer = PerturbationModelTrainer(**config)
    trainer.run(
        output_path=output_path,
        performance_per_perturbation=[True, False],
        metric=["spearman", "pearson", "mse"],
        should_load_data=True,
        save_test_predicted_labels=save_perf,
        save_trainer=save_trainer,
    )
    return trainer


# %%
def launch_jobs(model_id: str, task_id: str, rpz_random_state: int, output_path: str) -> None:
    """Launch jobs for a given model and task.

    Parameters
    ----------
    model_id : str
        The model id to use. Possible values are listed in the config_perturbation_model file.
    task_id : str
        The task id to use. Possible values are listed in the config_trainer file.
    rpz_random_state : int
        The random state to use.
    output_path : str
        The path to the output folder where the results are saved.
    """
    is_target_tissue_task = task_id[0] in ["3", "4"]

    # Launch multiple training jobs
    if is_target_tissue_task:
        # Get the list of target tissues for tasks in unseen tissues
        list_target_tissues = pd.read_csv(REPO_PATH / "data" / f"list_of_tissues_task_{task_id}.csv", header=None)[
            0
        ].tolist()

        # Loop over the target tissues (for tasks in unseen tissues only)
        task_id_without_tissue = task_id
        n_jobs = len(list_target_tissues)
        logger.info(f"Launching {n_jobs} jobs for target tissues...")
        for target_tissue in list_target_tissues:
            logger.info(f"Starting job for target tissue: {target_tissue}...")

            # Add tissue name to task id for tasks on target tissues
            if "3" in task_id or "4" in task_id:
                task_id = f"{task_id_without_tissue}_{rename_for_code(target_tissue)}"

            # Launch the training job for this tissue
            run_task(task_id=task_id, model_id=model_id, rpz_random_state=rpz_random_state, output_path=output_path)

    # Launch a single training job
    else:
        trainer = run_task(
            task_id=task_id, model_id=model_id, rpz_random_state=rpz_random_state, output_path=output_path
        )
        # Calculate performances in external validation set (PRISM) for task 2a
        if task_id == "2":
            # Define the experiment name
            experiment_name = f"task_{task_id}_model_{model_id}_seed_{rpz_random_state}"
            # Evaluate model performances in full prism dataset
            evaluate_in_prism(trainer=trainer, experiment_name=experiment_name, reproducibility_mask=False)
            # Evaluate label reproducibility and model performances in same prism data
            evaluate_in_prism(trainer=trainer, experiment_name=experiment_name, reproducibility_mask=True)


def evaluate_in_prism(
    trainer: PerturbationModelTrainer, experiment_name: str, reproducibility_mask: bool = False
) -> None:
    """Evaluate the model in PRISM."""
    # Load PRISM data
    config_data_prism = copy.deepcopy(trainer.config_source_domain_data)
    config_data_prism.studies = ["PRISM_2020"]
    config_data_prism.min_n_label = 15  # enough for test
    prism_data: PreclinicalDataset = instantiate(config_data_prism)

    if reproducibility_mask:
        # Keep cell lines that are available in both prism and pharmacodb
        prism_data.df_labels = prism_data.df_labels.loc[prism_data.df_labels.index.isin(trainer.data.df_labels.index)]
        prism_data.align_sample_data()
        pdb_data = copy.deepcopy(trainer.data)
        pdb_data.df_labels = pdb_data.df_labels.loc[
            prism_data.df_labels.index,
            prism_data.df_labels.columns,
        ]
        pdb_data.align_sample_data()

        # Keep drugs that are available in both prism and pharmacodb
        common_drugs = prism_data.df_labels.columns.intersection(pdb_data.df_labels.columns).tolist()
        prism_data.keep_perturbations(common_drugs)
        pdb_data.keep_perturbations(common_drugs)

        # Mask the predicted labels that are not available in pharmacodb
        # This is to ensure that, for each perturbation, the evaluation is done on the
        # same cell lines as in the prism reproducibility baseline.
        prism_data.df_labels = prism_data.df_labels.where(~pdb_data.df_labels.isna())

        # Mask the training set pairs for each split
        test_true_labels = _mask_by_split(prism_data.df_labels, trainer.split_pair_ids)
        test_predicted_labels = _mask_by_split(pdb_data.df_labels, trainer.split_pair_ids)

        # Remove any drug for which there are less than 15 cell lines
        test_true_labels, test_predicted_labels = _remove_drugs_with_few_samples(
            test_true_labels, test_predicted_labels
        )

        # Run the reproducibility baseline
        # The measured labels in pharmacodb are used as predicted values.
        reproducibility_performance = trainer.compute_performances(
            test_true_labels=test_true_labels, test_predicted_labels=test_predicted_labels
        )

        # Save reproducibility baseline performances in prism
        if trainer.output_path is not None:
            save_pickle(
                Path(trainer.output_path).joinpath("baseline_perf_repro_mask_prism.pkl"),
                reproducibility_performance,
            )
    else:
        # Mask the training set pairs for each split
        test_true_labels = _mask_by_split(prism_data.df_labels, trainer.split_pair_ids)

    # Predict labels in prism
    test_predicted_labels = trainer.predict(
        X=prism_data.df_rnaseq,
        X_metadata=prism_data.df_sample_metadata,
        X_fgpt=prism_data.df_fingerprints,
        refit_preprocessor=False,  # using the same trained preprocessor
        ensemble_over_splits=False,
    )

    # Remove any drug for which there are less than 15 cell lines
    (
        test_true_labels,
        test_predicted_labels,
    ) = _remove_drugs_with_few_samples(test_true_labels, test_predicted_labels)

    # Calculate the performances
    test_performance = trainer.compute_performances(
        test_true_labels=test_true_labels,
        test_predicted_labels=test_predicted_labels,
    )

    # Save predicted values and performances in prism
    if trainer.output_path is not None:
        save_pickle(
            Path(trainer.output_path).joinpath(
                f"{experiment_name}_pred_repro_mask_prism.pkl"
                if reproducibility_mask
                else f"{experiment_name}_pred_prism.pkl"
            ),
            test_predicted_labels,
        )
        save_pickle(
            Path(trainer.output_path).joinpath(
                "ground_truth_repro_mask_prism.pkl" if reproducibility_mask else "ground_truth_prism.pkl"
            ),
            test_true_labels,
        )
        save_pickle(
            Path(trainer.output_path).joinpath(
                f"{experiment_name}_perf_repro_mask_prism.pkl"
                if reproducibility_mask
                else f"{experiment_name}_perf_prism.pkl"
            ),
            test_performance,
        )


def _mask_by_split(df_labels: pd.DataFrame, split_pair_ids: dict[str, SplitPairIds]) -> dict[str, pd.DataFrame]:
    test_labels = {}
    for split in split_pair_ids:
        # Remove training set pairs from the labels
        df_labels_masked = df_labels.copy()
        for sample, perturbation in split_pair_ids[split]["training_ids"]:
            if sample in df_labels_masked.index and perturbation in df_labels_masked.columns:
                df_labels_masked.loc[sample, perturbation] = np.nan
        test_labels[split] = df_labels_masked
    return test_labels


def _remove_drugs_with_few_samples(
    test_true_labels: dict[str, pd.DataFrame], test_predicted_labels: dict[str, pd.DataFrame], min_n_label: int = 15
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Remove any drug for which there are less than min_n_label cell lines."""
    list_drugs_to_remove = []
    for df_labels in test_true_labels.values():
        n_samples_per_drug = len(df_labels) - df_labels.isna().sum(axis=0)
        list_drugs_to_remove.extend(n_samples_per_drug[n_samples_per_drug < min_n_label].index.tolist())
    list_drugs_to_remove = list(set(list_drugs_to_remove))
    for split in test_true_labels:
        test_true_labels[split] = test_true_labels[split].loc[
            :, ~test_true_labels[split].columns.isin(list_drugs_to_remove)
        ]
        test_predicted_labels[split] = test_predicted_labels[split].loc[
            :, ~test_predicted_labels[split].columns.isin(list_drugs_to_remove)
        ]
    logger.info(f"Number of drugs removed: {len(list_drugs_to_remove)}")
    return test_true_labels, test_predicted_labels


# %%
if __name__ == "__main__":
    parsed_args = get_task_parser()
    launch_jobs(**vars(parsed_args))
