"""Ensemble over representation-specific regression models for a given task."""

# %%
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from configs.config_perturbation_model import PRED_MODEL_NAME
from leap.trainer.perturbation_model_trainer import PerturbationModelTrainer
from leap.utils.io import load_pickle, save_pickle

REPO_PATH = Path(__file__).parent.parent


# %%
def _get_latest_result_dir(task_id: str, model_id: str, rpz_random_state: int) -> Path | None:
    """Get the latest result directory for a given task, model, and seed.

    Parameters
    ----------
    task_id : str
        The task identifier.
    model_id : str
        The model identifier.
    rpz_random_state : int
        The random seed for the representation model.

    Returns
    -------
    Path | None
        Path to the latest result directory, or None if not found.
    """
    result_dir = REPO_PATH / "results" / f"task_{task_id}_model_{model_id}_seed_{rpz_random_state}"
    if not result_dir.exists():
        return None

    # Find the latest timestamped subdirectory
    timestamped_dirs = sorted(result_dir.iterdir())
    return timestamped_dirs[-1] if timestamped_dirs else None


def _get_result_file_path(task_id: str, model_id: str, rpz_random_state: int, file_type: str = "perf") -> Path | None:
    """Get the path to a result file (perf, pred, trainer, etc.).

    Parameters
    ----------
    task_id : str
        The task identifier.
    model_id : str
        The model identifier.
    rpz_random_state : int
        The random seed for the representation model.
    file_type : str, optional
        The type of file to retrieve ("perf", "pred", "trainer"), by default "perf".

    Returns
    -------
    Path | None
        Path to the result file, or None if not found.
    """
    result_dir = _get_latest_result_dir(task_id, model_id, rpz_random_state)
    if result_dir is None:
        return None

    file_name = f"task_{task_id}_model_{model_id}_seed_{rpz_random_state}_{file_type}.pkl"
    file_path = result_dir / file_name
    return file_path if file_path.exists() else None


def _check_all_seeds_exist(task_id: str, model_id: str, n_seeds: int = 5) -> bool:
    """Check if result files exist for all seeds.

    Parameters
    ----------
    task_id : str
        The task identifier.
    model_id : str
        The model identifier.
    n_seeds : int, optional
        Number of seeds to check, by default 5.

    Returns
    -------
    bool
        True if all seeds have results, False otherwise.
    """
    for seed in range(n_seeds):
        if _get_result_file_path(task_id, model_id, seed, "perf") is None:
            logger.warning(f"Missing results for seed {seed}")
            return False
    return True


def _load_and_average_predictions(
    task_id: str,
    model_id: str,
    file_type: str = "pred",
    n_seeds: int = 5,
) -> dict:
    """Load and average predictions from multiple representation seeds.

    Parameters
    ----------
    task_id : str
        The task identifier.
    model_id : str
        The model identifier.
    file_type : str, optional
        The type of prediction file ("pred", "pred_prism", etc.), by default "pred".
    n_seeds : int, optional
        Number of seeds to ensemble, by default 5.

    Returns
    -------
    dict
        Dictionary with split names as keys and averaged predictions as values.

    Raises
    ------
    FileNotFoundError
        If prediction files are missing for any seed.
    """
    ensemble_predictions: dict | None = None

    for seed in range(n_seeds):
        file_path = _get_result_file_path(task_id, model_id, seed, file_type)
        if file_path is None:
            raise FileNotFoundError(f"Missing {file_type} file for task={task_id}, model={model_id}, seed={seed}")

        predictions = load_pickle(file_path)

        if ensemble_predictions is None:
            # Initialize with first seed
            ensemble_predictions = predictions
        else:
            # Add predictions from subsequent seeds
            for split in predictions:
                ensemble_predictions[split] += predictions[split]

    # Type narrowing: we know ensemble_predictions is not None after the loop
    assert ensemble_predictions is not None, "No predictions loaded"

    # Average across seeds
    for split in ensemble_predictions:
        ensemble_predictions[split] /= n_seeds

    return ensemble_predictions


def save_representation_ensemble(
    task_id: str,
    model_id: str,
    n_seeds: int = 5,
) -> None:
    """Ensemble predictions from multiple representation seeds and save results.

    Loads predictions from multiple representation model seeds, averages them,
    computes ensemble performance, and saves results.

    Parameters
    ----------
    task_id : str
        The task identifier.
    model_id : str
        The model identifier.
    n_seeds : int, optional
        Number of representation seeds to ensemble, by default 5.

    Raises
    ------
    FileNotFoundError
        If required result files are not found.
    """
    logger.info(f"Creating ensemble for task={task_id}, model={model_id}, n_seeds={n_seeds}")

    # Check all seeds exist
    if not _check_all_seeds_exist(task_id, model_id, n_seeds):
        raise FileNotFoundError(f"Missing results for some seeds in task={task_id}, model={model_id}")

    # Load trainer from seed 0 (for computing metrics and getting ground truth)
    trainer_path = _get_result_file_path(task_id, model_id, rpz_random_state=0, file_type="trainer")
    if trainer_path is None:
        raise FileNotFoundError(f"Trainer file not found for task={task_id}, model={model_id}, seed=0")
    trainer: PerturbationModelTrainer = load_pickle(trainer_path)

    # Sanity check: verify saved performance matches recomputed performance for seed 0
    logger.info("Running sanity check on seed 0...")
    saved_perf = load_pickle(_get_result_file_path(task_id, model_id, 0, "perf"))
    recomputed_perf, _ = trainer.evaluate(
        test_true_labels=trainer.get_test_true_labels(data=trainer.data, split_pair_ids=trainer.split_pair_ids),
        test_predicted_labels=trainer.test_predicted_labels,
    )
    saved_score = pd.DataFrame(saved_perf["per_perturbation"]["spearman"]).mean().mean()
    recomputed_score = pd.DataFrame(recomputed_perf["per_perturbation"]["spearman"]).mean().mean()
    assert round(saved_score, 4) == round(recomputed_score, 4), "Sanity check failed: scores don't match!"
    logger.info("Sanity check passed")

    # Create ensemble predictions by averaging across seeds
    logger.info("Creating ensemble predictions...")
    ensemble_predictions = _load_and_average_predictions(task_id, model_id, file_type="pred", n_seeds=n_seeds)

    # Compute ensemble performance
    logger.info("Computing ensemble performance...")
    ground_truth = trainer.get_test_true_labels(data=trainer.data, split_pair_ids=trainer.split_pair_ids)
    ensemble_performance = trainer.compute_performances(
        test_true_labels=ground_truth,
        test_predicted_labels=ensemble_predictions,
    )

    # Log the ensemble performance
    trainer.log_performance_metrics(ensemble_performance)

    # Save ensemble results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = REPO_PATH / "results" / f"task_{task_id}_model_{model_id}_ensemble_seed_0" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    save_pickle(ensemble_predictions, save_dir / f"task_{task_id}_model_{model_id}_ensemble_pred.pkl")
    save_pickle(ensemble_performance, save_dir / f"task_{task_id}_model_{model_id}_ensemble_perf.pkl")
    logger.info(f"Ensemble results saved to {save_dir}")

    # Handle PRISM evaluation for task 2
    if task_id == "2":
        logger.info("Processing PRISM evaluations...")
        for prism_type in ("prism", "repro_mask_prism"):
            # Ensemble PRISM predictions
            prism_predictions = _load_and_average_predictions(
                task_id, model_id, file_type=f"pred_{prism_type}", n_seeds=n_seeds
            )

            # Load ground truth (from seed 0 results)
            seed0_dir = _get_latest_result_dir(task_id, model_id, 0)
            if seed0_dir is None:
                logger.warning(f"Seed 0 directory not found for {prism_type}, skipping")
                continue
            ground_truth_file = seed0_dir / f"ground_truth_{prism_type}.pkl"
            if not ground_truth_file.exists():
                logger.warning(f"Ground truth file not found for {prism_type}, skipping")
                continue
            prism_ground_truth = load_pickle(ground_truth_file)

            # Compute and save performance
            prism_performance = trainer.compute_performances(
                test_true_labels=prism_ground_truth,
                test_predicted_labels=prism_predictions,
            )

            save_pickle(prism_predictions, save_dir / f"task_{task_id}_model_{model_id}_ensemble_pred_{prism_type}.pkl")
            save_pickle(prism_performance, save_dir / f"task_{task_id}_model_{model_id}_ensemble_perf_{prism_type}.pkl")
            logger.info(f"PRISM {prism_type} results saved")


# %%
def get_ensemble_parser() -> argparse.Namespace:
    """Argument parser for the pred_task function."""
    # Simulate passing arguments if running the script as a notebook
    if "3_ensemble_representation_layers.py" not in sys.argv[0]:
        sys.argv = ["3_ensemble_representation_layers.py"]

    # Parse arguments
    parser = argparse.ArgumentParser(description="Ensemble over representation-specific regression models.")
    parser.add_argument("--task_id", type=str, default="1")
    parser.add_argument("--model_id", type=str, choices=PRED_MODEL_NAME.keys(), default="mae_ps_enet")
    args = parser.parse_args()
    return args


def _normalize_tissue_name(tissue: str) -> str:
    """Convert tissue name to standardized format for file paths."""
    return tissue.replace("/", "_").replace(" ", "_").lower()


def _ensemble_already_exists(task_id: str, model_id: str) -> bool:
    """Check if ensemble results already exist for a task and model."""
    ensemble_dir = REPO_PATH / "results" / f"task_{task_id}_model_{model_id}_ensemble_seed_0"
    return ensemble_dir.exists() and any(p.is_file() for p in ensemble_dir.iterdir())


def ensemble_representation_layers(task_id: str, model_id: str, n_seeds: int = 5) -> None:
    """Ensemble predictions from multiple representation seeds for a task.

    For tissue-specific tasks (task 3 or 4), ensembles each tissue separately.
    For other tasks, creates a single ensemble.

    Parameters
    ----------
    task_id : str
        The task identifier (e.g., "1", "2", "3", "4").
    model_id : str
        The model identifier.
    n_seeds : int, optional
        Number of representation seeds to ensemble, by default 5.
    """
    logger.info(f"Starting ensemble for task={task_id}, model={model_id}")

    # Handle tissue-specific tasks (3 and 4)
    if task_id[0] in ["3", "4"]:
        tissues_file = REPO_PATH / "data" / f"list_of_tissues_task_{task_id}.csv"
        tissues = pd.read_csv(tissues_file, header=None)[0].tolist()
        tissues = [_normalize_tissue_name(t) for t in tissues]

        logger.info(f"Processing {len(tissues)} tissues for task {task_id}")
        for tissue in tissues:
            task_tissue = f"{task_id}_{tissue}"

            # Skip if ensemble already exists
            if _ensemble_already_exists(task_tissue, model_id):
                logger.info(f"Ensemble already exists for tissue={tissue}, skipping")
                continue

            # Check if all seeds exist
            if not _check_all_seeds_exist(task_tissue, model_id, n_seeds):
                logger.warning(f"Missing results for tissue={tissue}, skipping")
                continue

            # Create ensemble
            try:
                save_representation_ensemble(task_tissue, model_id, n_seeds)
                logger.info(f"Successfully ensembled tissue={tissue}")
            except Exception as e:
                logger.error(f"Failed to ensemble tissue={tissue}: {e}")
    else:
        # Handle regular tasks
        if _ensemble_already_exists(task_id, model_id):
            logger.info(f"Ensemble already exists for task={task_id}, skipping")
            return

        save_representation_ensemble(task_id, model_id, n_seeds)


# %%
if __name__ == "__main__":
    parsed_args = get_ensemble_parser()
    ensemble_representation_layers(**vars(parsed_args))
