"""Trainer for LEAP."""

import copy
import os
import subprocess
import time
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
import psutil
import torch
from loguru import logger
from ml_collections import config_dict

from leap.data.preclinical_dataset import PreclinicalDataset
from leap.metrics.regression_metrics import RegressionMetricType, performance_metric_wrapper
from leap.pipelines.perturbation_pipeline import PerturbationPipeline
from leap.utils.config_utils import instantiate
from leap.utils.io import save_pickle


class SplitPairIds(TypedDict):
    """Dictionary of sample x perturbation pairs in each split."""

    training_ids: list[tuple[Any, Any]]
    test_ids: list[tuple[Any, Any]]


class SplitIds(TypedDict):
    """Dictionary of sample ids in each split."""

    training_ids: list
    test_ids: list


def check_list_pair(value: Any) -> list[tuple[Any, Any]]:
    """Check that the input is a list of pairs."""
    if not isinstance(value, list):
        raise ValueError(f"Expected a list, got {type(value)}")
    if not all(isinstance(pair, tuple) for pair in value):
        raise ValueError(f"Expected a list of pairs, got {value}")
    if not all(len(pair) == 2 for pair in value):
        raise ValueError(f"Expected a list of pairs, got {value}")
    return cast(list[tuple[Any, Any]], value)


class PerturbationModelTrainer:
    """Trainer for perturbation prediction models.

    This class manages the complete pipeline for training and evaluating perturbation prediction models, including data
    splitting, model training, prediction, and performance evaluation.

    Parameters
    ----------
    source_domain_data : config_dict.ConfigDict
        The source domain data configuration.
    data_split : config_dict.ConfigDict
        The split configuration for source domain data.
    model : config_dict.ConfigDict
        The model configuration.
    target_domain_data : config_dict.ConfigDict | None
        The target domain data configuration, optional.
    """

    def __init__(
        self,
        source_domain_data: config_dict.ConfigDict,
        data_split: config_dict.ConfigDict,
        model: config_dict.ConfigDict,
        target_domain_data: config_dict.ConfigDict | None = None,
    ):
        # Store configurations
        self.config_source_domain_data = source_domain_data
        self.config_data_split = data_split
        self.config_model = model
        self.config_target_domain_data = target_domain_data
        self._output_path: Path | None = None

        # Initialize other attributes
        self._data: PreclinicalDataset | None = None
        self.split_pair_ids: dict[str, SplitPairIds] = {}
        self.trained_model: dict[str, PerturbationPipeline] = {}
        self.test_predicted_labels: dict[Any, pd.DataFrame] = {}
        self.test_performance: dict[str, dict[str, dict[Any, dict[str, float]]]] = {}
        self.test_performance_aggregated: dict[str, dict[str, dict[str, str]]] = {}
        self.run_time: str = "0"

        # Store n cpus, n gpus and ram to keep context of run time estimation
        try:
            self.n_cpus = subprocess.run("nproc", capture_output=True, text=True, check=True).stdout.strip()
        except FileNotFoundError:
            self.n_cpus = subprocess.run(
                ["sysctl", "-n", "hw.logicalcpu"], capture_output=True, text=True, check=True
            ).stdout.strip()
        self.n_gpus = torch.cuda.device_count()
        try:
            self.ram = subprocess.run(
                "free -h | grep Mem | tr -s ' ' | cut -d ' ' -f 2",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            # macOS doesn't have 'free' command
            self.ram = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            ).stdout.strip()
            # Convert bytes to GB for consistency with Linux output
            self.ram = f"{int(self.ram) / (1024**3):.1f}G"

    @property
    def output_path(self) -> Path:
        """Return the output path."""
        if self._output_path is None:
            raise ValueError("Output path is not set")
        return self._output_path

    @output_path.setter
    def output_path(self, output_path: Path) -> None:
        """Set the output path."""
        self._output_path = output_path

    @property
    def data(self) -> PreclinicalDataset:
        """Return the source and target domain data.

        Note: This is a property to ensure data is loaded before access.
        """
        if self._data is None:
            raise ValueError("Data is not loaded")
        return self._data

    @data.setter
    def data(self, data: PreclinicalDataset) -> None:
        """Set the source and target domain data."""
        self._data = data

    def __setstate__(self, state: dict[str, Any]) -> None:
        if "data" in state:
            state["_data"] = state.pop("data")
        if "output_path" in state:
            state["_output_path"] = state.pop("output_path")
        self.__dict__.update(state)

    def __str__(self) -> str:
        description = """Class attributes:

        Configurations:
        config_source_domain_data: source domain data configuration
        config_data_split: split configuration in source domain data
        config_model: model configuration
        config_target_domain_data: target domain data configuration

        Results:
        data: source and target domain data
        split_pair_ids: dictionary of sample x perturbation pairs in each split
        trained_model: trained prediction model
        test_predicted_labels: predictions on the test set(s)
        test_performance: performance on the test set(s)
        test_performance_aggregated: aggregated performance on the test set(s)
        run_time: time to run the pipeline (in seconds)
        n_cpus: total number of cpus where the trainer is instantiated
        n_gpus: total number of gpus where the trainer is instantiated
        ram: total ram where the trainer is instantiated
        """
        return description

    def run(
        self,
        output_path: str = "",
        performance_per_perturbation: bool | list[bool] | None = None,
        metric: RegressionMetricType | list[RegressionMetricType] | None = None,
        should_load_data: bool = True,
        save_test_predicted_labels: bool = False,
        save_trainer: bool = True,
        start_split_n: int = -1,
        n_splits: int = -1,
    ) -> None:
        """Run the pipeline.

        There are 5 steps:
        1. Load the data (optional).
        2. Split the data into training and test sets.
        3. Train the model (including hyper-parameter tuning in CV).
        4. Predict labels in the test set(s).
        5. Evaluate prediction performance in the test set(s).
        6. Save the results if output_path is provided. The prediction performances in the test set(s) are saved in a
        .pkl file. The predicted labels and the entire trainer can also be saved in separate .pkl files.

        Parameters
        ----------
        output_path : str, optional
            Path to the output folder where the results are saved. This path must end with experiment_name / date_time.
            The final instance of the class, predicted labels and performances are pickled and saved in the specified
            folder. The logs are also saved in the same folder. The results are not saved if the output_path is "".
        performance_per_perturbation : bool | list[bool] | None
            Whether to evaluate performance per perturbation (True) or over all perturbations (False). By default, both
            metrics are computed.
        metric : RegressionMetricType | list[RegressionMetricType] | None
            The metric(s) to use for evaluation. Possible values are: "spearman", "pearson", "r2", "mse" and "mae". By
            default, all metrics are computed.
        should_load_data : bool, optional
            Whether to load the data. The data can be loaded separately in a script for more complex cases (for
            instance, for the DDT benchmark), by default True.
        save_test_predicted_labels : bool, optional
            Whether to save the predicted labels in the test set(s), by default False.
        save_trainer : bool, optional
            Whether to save the final instance of the class, by default True.
        start_split_n : int, optional
            The split number to start from, by default -1 to start from the first split.
        n_splits : int, optional
            The number of splits to keep, by default -1 to run all splits.
        """
        # Define arguments
        if performance_per_perturbation is None:
            performance_per_perturbation = [True, False]
        if metric is None:
            metric = ["spearman", "pearson", "r2", "mse", "mae"]

        # Prepare output paths
        if output_path != "":
            # Create a folder with the following structure:
            # output_path/
            #   experiment_name/
            #       date_time/
            #         experiment_name_perf.pkl
            #         experiment_name_pred.pkl
            #         experiment_name_trainer.pkl
            #         saved_ensembles/
            #           split_id/
            #             model_perturbation_A_id.pkl
            #             model_perturbation_B_id.pkl
            #             ...

            # Extract the experiment_name from the output path provided
            experiment_name = Path(output_path).parts[-2]
            self.output_path = Path(output_path)

            # Define the file names
            log_file = self.output_path.joinpath(f"{experiment_name}.log")
            data_summary_file = self.output_path.joinpath(f"{experiment_name}_data_summary.csv")
            perf_pkl_file = self.output_path.joinpath(f"{experiment_name}_perf.pkl")
            pred_pkl_file = self.output_path.joinpath(f"{experiment_name}_pred.pkl")
            trainer_pkl_file = self.output_path.joinpath(f"{experiment_name}_trainer.pkl")

            # Start a new log file
            logger.add(log_file)
            logger.info(f"Starting pipeline with output path: {perf_pkl_file}")

        # Load the data
        if should_load_data:
            self.data = self.load_data()

        # Log and save data summary
        self.log_data_summary(data=self.data)
        if output_path != "":
            self.save_data_summary(data=self.data, output_path=data_summary_file)

        # Split the data into training and test sets
        self.split_pair_ids = self.split_training_test(data=self.data)

        # if provided, keep only n_splits splits starting from start_split_n
        if n_splits > 0:
            logger.info(f"Keeping only {n_splits} splits")
            if start_split_n < 0:
                logger.warning("As start_split_n was left unspecified, it was set to 0.")
                start_split_n = 0
            self.split_pair_ids = self._keep_n_splits(
                split_pair_ids=self.split_pair_ids, start_split_n=start_split_n, n_splits=n_splits
            )

        # Train models for each of the training sets.
        self.train(data=self.data, split_pair_ids=self.split_pair_ids)

        # Evaluate performances in the test sets
        self.test_predicted_labels = self.predict_test(data=self.data, split_pair_ids=self.split_pair_ids)
        self.test_performance, self.test_performance_aggregated = self.evaluate(
            test_true_labels=self.get_test_true_labels(data=self.data, split_pair_ids=self.split_pair_ids),
            test_predicted_labels=self.test_predicted_labels,
            performance_per_perturbation=performance_per_perturbation,
            metric=metric,
        )

        # Save the predicted values and performances
        if output_path != "":
            self.save_attribute("test_performance", perf_pkl_file)
            if save_test_predicted_labels:
                self.save_attribute("test_predicted_labels", pred_pkl_file)

        # Save the trainer
        if output_path != "" and save_trainer:
            self.save_trainer(trainer_pkl_file)

    def _keep_n_splits(
        self, split_pair_ids: dict[str, SplitPairIds], start_split_n: int, n_splits: int
    ) -> dict[str, SplitPairIds]:
        """Keep only n_splits splits starting from start_split_n."""
        # Check that there are at least n_splits + start_split_n splits
        if start_split_n + n_splits > len(split_pair_ids):
            raise ValueError(f"Not enough splits to keep {n_splits} starting from {start_split_n}")
        # Check that start_split_n is valid
        if start_split_n < 0 or start_split_n >= len(split_pair_ids):
            raise ValueError(f"Invalid start split number: {start_split_n}")
        logger.info(f"Keeping only {n_splits} splits starting from {start_split_n}")
        return dict(list(split_pair_ids.items())[start_split_n : start_split_n + n_splits])

    def load_data(self) -> PreclinicalDataset:
        """Load the data based on the source and target domain configurations.

        Returns
        -------
        PreclinicalDataset
            The full dataset, including source and target domains (if available).
        """
        dataset: PreclinicalDataset = instantiate(self.config_source_domain_data)
        dataset.df_sample_metadata["domain"] = "source"
        dataset.stack_dataframes()
        if self.config_target_domain_data is not None:
            # Load target domain data if available
            dataset_target: PreclinicalDataset = instantiate(self.config_target_domain_data)
            dataset_target.df_sample_metadata["domain"] = "target"
            dataset_target.stack_dataframes()

            # Keep common perturbations between source and target domains
            perturbation_names = dataset.df_labels.columns.intersection(dataset_target.df_labels.columns).tolist()
            dataset.keep_perturbations(perturbation_names)
            dataset_target.keep_perturbations(perturbation_names)

            # Keep common columns between source and target domains
            columns_name = dataset.df_rnaseq.columns.intersection(dataset_target.df_rnaseq.columns).tolist()
            if len(columns_name) < len(dataset.df_rnaseq.columns):
                logger.warning(
                    f"Dropping {len(dataset.df_rnaseq.columns) - len(columns_name)} genes that"
                    " are not common between source and target domains."
                )
            dataset.df_rnaseq = dataset.df_rnaseq[columns_name]
            dataset_target.df_rnaseq = dataset_target.df_rnaseq[columns_name]

            # Keep only samples with at least one label for those common perturbations
            dataset_target.df_labels.dropna(how="all", inplace=True)
            dataset_target.align_sample_data()
            dataset.df_labels.dropna(how="all", inplace=True)
            dataset.align_sample_data()
            # Concatenate source and target domain data
            dataset.merge(dataset_target)
            dataset._sort_rows_and_columns()

        return dataset

    def log_data_summary(self, data: PreclinicalDataset) -> None:
        """Log number of genes, perturbations and samples in the data.

        Parameters
        ----------
        data : PreclinicalDataset
            The full dataset, including source and target domains (if available).
        """
        message = (
            f"Data summary: "
            f"{len(data.df_labels.columns)} perturbations, "
            f"{len(data.df_labels)} samples, "
            f"{data.df_labels.count().sum()} unique pairs, "
            f"{len(data.df_rnaseq.columns)} genes"
        )
        logger.info(message)

    def save_data_summary(self, data: PreclinicalDataset, output_path: Path) -> None:
        """Save the data summary."""
        # Initialise dictionary to store counts
        count_dict = []
        index = []

        # Count the number of genes, perturbations and samples
        count_dict.append(len(data.df_rnaseq.columns))
        index.append("Number of genes")
        count_dict.append(len(data.df_labels))
        index.append("Number of samples")
        count_dict.append(len(data.df_labels.columns))
        index.append("Number of perturbations")
        count_dict.append(data.df_labels.count().sum())
        index.append("Number of sample x perturbation pairs")

        # Save the data summary
        pd.DataFrame(
            count_dict,
            index=index,
        ).transpose().to_csv(
            str(output_path),
            index=True,
            header=True,
        )

    def split_training_test(self, data: PreclinicalDataset) -> dict[str, SplitPairIds]:
        """Create training and test splits.

        The indices of samples x perturbation pairs in:
            - the training set: source domain samples only
            - the test set: target domain samples only (if available), otherwise source domain samples only

        Parameters
        ----------
        data : PreclinicalDataset
            The full dataset, including source and target domains (if available).

        Returns
        -------
        dict[str, SplitPairIds]
            The dictionary of sample x perturbation pairs in each split. The keys are the split ids and values are typed
            dictionaries with keys "training_ids" and "test_ids".
        """
        # Initialise dictionary to store split ids
        split_pair_ids: dict[str, SplitPairIds] = {}

        if "target" in data.df_sample_metadata["domain"].tolist():
            # If a target domain is provided: the full source domain data is used as training set and the target domain
            # data is split into a test set.

            # Define training set as full source domain data
            training_ids = check_list_pair(
                data.df_sample_metadata_stacked.loc[
                    data.df_sample_metadata_stacked["domain"] == "source"
                ].index.to_list()
            )

            # Create splits of sample x perturbation pairs from target domain
            split_data = instantiate(self.config_data_split)
            split_ids_dict = split_data(
                X_metadata=data.df_sample_metadata_stacked.loc[data.df_sample_metadata_stacked["domain"] == "target"]
            )

            # Store all split ids
            for split in split_ids_dict:
                split_pair_ids[split] = SplitPairIds(
                    training_ids=training_ids,
                    test_ids=check_list_pair(split_ids_dict[split]["test_ids"]),
                )
        else:
            # If only a source domain is provided: it is split into a training set and a test set.
            # Create splits of sample x perturbation pairs from source domain
            split_data = instantiate(self.config_data_split)
            split_ids_dict = split_data(
                X_metadata=data.df_sample_metadata_stacked.loc[data.df_sample_metadata_stacked["domain"] == "source"]
            )

            # Store all split ids
            for split in split_ids_dict:
                split_pair_ids[split] = SplitPairIds(
                    training_ids=check_list_pair(split_ids_dict[split]["training_ids"]),
                    test_ids=check_list_pair(split_ids_dict[split]["test_ids"]),
                )

        return split_pair_ids

    def extract_split_sample_ids(self, split_pair_ids: dict[str, SplitPairIds]) -> dict[Any, SplitIds]:
        """Extract sample ids in each split.

        Parameters
        ----------
        split_pair_ids : dict[str, SplitPairIds]
            The dictionary of sample x perturbation pairs in each split.

        Returns
        -------
        dict[Any, SplitIds]
            Dictionary of sample ids in each split.
        """
        return self._loop_over_ids(split_pair_ids, pair_id=0)

    def extract_split_perturbation_ids(self, split_pair_ids: dict[str, SplitPairIds]) -> dict[Any, SplitIds]:
        """Extract perturbation ids in each split.

        Parameters
        ----------
        split_pair_ids : dict[str, SplitPairIds]
            The dictionary of sample x perturbation pairs in each split.

        Returns
        -------
        dict[Any, SplitIds]
            Dictionary of perturbation ids in each split.
        """
        return self._loop_over_ids(split_pair_ids, pair_id=1)

    def _loop_over_ids(self, split_pair_ids: dict[str, SplitPairIds], pair_id: int = 0) -> dict[Any, SplitIds]:
        """Loop over sample x perturbation pairs to extract sample/perturbation ids."""
        split_ids: dict[Any, SplitIds] = {}
        for split, split_dict in split_pair_ids.items():
            split_ids[split] = SplitIds(
                training_ids=self._get_pair_member(pair_list=split_dict["training_ids"], pair_id=pair_id),
                test_ids=self._get_pair_member(pair_list=split_dict["test_ids"], pair_id=pair_id),
            )

        return split_ids

    @staticmethod
    def _get_pair_member(pair_list: list[tuple[Any, Any]], pair_id: int) -> list:
        if pair_id not in {0, 1}:
            raise ValueError("pair_id must be 0 or 1")
        return list(np.unique([pair[pair_id] for pair in pair_list]))

    def train(self, data: PreclinicalDataset, split_pair_ids: dict[str, SplitPairIds]) -> None:
        """Train the model.

        Parameters
        ----------
        data : PreclinicalDataset
            The full dataset, including source and target domains (if available).
        split_pair_ids : dict[str, SplitPairIds]
            The dictionary of sample x perturbation pairs in each split.
        """
        start_time = time.time()
        # Initialise dictionary to store trained models and save first split name
        for split, split_pair_id in split_pair_ids.items():
            n_splits = len(split_pair_ids)
            logger.info(f"Training model, {split.replace('_', ' ')} over {n_splits}...")

            # Extract fingerprints
            X_fgpt = data.df_fingerprints

            # Extract training data
            X, y, X_metadata = self._get_training_data(data, split_pair_id)

            # The RegressionModel is instantiated from the config.
            self.trained_model[split] = copy.deepcopy(instantiate(self.config_model))

            # Add the ensembling output path to the model if provided
            if self._output_path is not None:
                ensembling_output_path = self.output_path.joinpath(f"saved_model_cv_ensembling_{split}")
                self.trained_model[split].ensembling_output_path = ensembling_output_path

            # Fit the model on the training data
            self.trained_model[split].fit(X=X, y=y.loc[X.index], X_fgpt=X_fgpt, X_metadata=X_metadata.loc[X.index])

            # Log memory usage
            process = psutil.Process(os.getpid())
            logger.info(
                f"Memory usage at the end of {split.replace('_', ' ')}: {process.memory_info().rss / (1024**3):.2f} GB"
            )

        end_time = time.time()
        self.run_time = f"{end_time - start_time:.0f}"
        logger.info(f"Training time: {self.run_time} seconds")

    def _get_training_data(
        self, data: PreclinicalDataset, current_split_pair_ids: SplitPairIds
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract the training data for a given split.

        Parameters
        ----------
        data : PreclinicalDataset
            The full dataset, including source and target domains (if available).
        current_split_pair_ids : SplitPairIds
            The dictionary of sample x perturbation pairs in the current split.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            The training input data (X), training label data (y), and training metadata (X_metadata).
        """
        # Define sample x perturbation pairs in the training set
        training_pair_ids = current_split_pair_ids["training_ids"]

        # Extract sample ids from the sample x perturbation pairs in the training set
        training_sample_ids = self._get_pair_member(training_pair_ids, 0)

        # Define the training label data
        # The labels are unstacked to go back to a dataframe where rows are samples and columns are perturbations. The
        # training/test split is created here by replacing some of the sample x perturbation pairs by NaN. The NaN
        # values in label data are filtered out internally when calling fit below. When using a grouping by sample in
        # self.split_training_test, then this is equivalent to a training/test split by sample. However, this framework
        # is more general as it allows for perturbation-specific training/test splits. This is needed for drug response
        # as a given drug has not been tested on all samples (it ensures that the same training/test set proportions are
        # used across all perturbations).
        y = data.df_labels_stacked.loc[training_pair_ids]["label"].unstack()
        # Extract RNASeq data
        X = data.df_rnaseq.loc[training_sample_ids]
        # Define the training metadata
        X_metadata = data.df_sample_metadata.loc[training_sample_ids]

        return (
            X,
            y,
            X_metadata,
        )

    def get_test_true_labels(
        self, data: PreclinicalDataset, split_pair_ids: dict[str, SplitPairIds]
    ) -> dict[Any, pd.DataFrame]:
        """Get labels in the test set(s).

        Parameters
        ----------
        data : PreclinicalDataset
            The full dataset, including source and target domains (if available).
        split_pair_ids : dict[str, SplitPairIds]
            The dictionary of sample x perturbation pairs in each split.

        Returns
        -------
        dict[Any, pd.DataFrame]
            Dictionary of labels in the test set(s).
        """
        test_true_labels = {}
        for split in split_pair_ids:
            # Define the test label data
            y_stacked_test = data.df_labels_stacked.loc[split_pair_ids[split]["test_ids"]]["label"]
            y_test = y_stacked_test.unstack()

            # Store the test labels
            test_true_labels[split] = y_test

        return test_true_labels

    def predict_test(
        self, data: PreclinicalDataset, split_pair_ids: dict[str, SplitPairIds]
    ) -> dict[Any, pd.DataFrame]:
        """Predict labels in test data.

        Parameters
        ----------
        data : PreclinicalDataset
            The full dataset, including source and target domains (if available).
        split_pair_ids : dict[str, SplitPairIds]
            The dictionary of sample x perturbation pairs in each split.

        Returns
        -------
        dict[Any, pd.DataFrame]
            Dictionary of predicted labels in the test set(s).

        Raises
        ------
        ValueError
            If there is no test set to predict.
        """
        # Predict labels if any of the test sets is not empty
        if any(len(split_dict["test_ids"]) > 0 for split_dict in split_pair_ids.values()):
            split_sample_ids = self.extract_split_sample_ids(split_pair_ids)
            split_perturbation_ids = self.extract_split_perturbation_ids(split_pair_ids)
            test_predicted_labels = {}
            for split, split_dict in split_sample_ids.items():
                # Fingerprints are always loaded in the trainer but only used by some
                # models
                X_fgpt = data.df_fingerprints
                X = data.df_rnaseq.loc[split_dict["test_ids"]]
                test_predicted_labels[split] = self.trained_model[split].predict(
                    X=X,
                    X_metadata=data.df_sample_metadata.loc[split_dict["test_ids"]],
                    X_fgpt=X_fgpt,
                    # Predict only for perturbations in the test set
                    list_of_perturbations=split_perturbation_ids[split]["test_ids"],
                )

            return test_predicted_labels
        raise ValueError("No test set to predict")

    def evaluate(
        self,
        test_true_labels: dict[Any, pd.DataFrame],
        test_predicted_labels: dict[Any, pd.DataFrame],
        performance_per_perturbation: bool | list[bool] | None = None,
        metric: RegressionMetricType | list[RegressionMetricType] | None = None,
    ) -> tuple[
        dict[str, dict[str, dict[Any, dict[str, float]]]],
        dict[str, dict[str, dict[str, str]]],
    ]:
        """Evaluate prediction performance in test data.

        Parameters
        ----------
        test_true_labels : dict[Any, pd.DataFrame]
            Dictionary of true labels in the test set(s).
        test_predicted_labels : dict[Any, pd.DataFrame]
            Dictionary of predicted labels in the test set(s).
        performance_per_perturbation : bool | list[bool] | None
            Whether to evaluate performance per perturbation (True) or over all perturbations (False). By default, both
            per perturbation and overall performances are calculated.
        metric : RegressionMetricType | list[RegressionMetricType] | None
            The metric(s) to use for evaluation. Possible values are: "spearman", "pearson", "r2", "mse" and "mae". By
            default, all possible values are calculated.

        Returns
        -------
        tuple[dict[str, dict[str, dict[Any, dict[str, float]]]], dict[str, dict[str, dict[str, str]]]]
            The dictionary of performances per split, per metric and per perturbation
            (mapping perturbation, metric and split)
            (if applicable) and the dictionary of aggregated performances over splits
            and labels.
        """
        logger.info("Evaluating prediction performances...")

        # Define arguments
        if performance_per_perturbation is None:
            performance_per_perturbation = [True, False]
        if metric is None:
            metric = ["spearman", "pearson", "r2", "mse", "mae"]
        # Calculate the performances
        test_performance = self.compute_performances(
            test_true_labels=test_true_labels,
            test_predicted_labels=test_predicted_labels,
            performance_per_perturbation=performance_per_perturbation,
            metric=metric,
        )

        # Aggregate the performances over splits and labels

        # Since format_numbers is true, the values are formatted to 4 decimal places
        test_performance_aggregated = cast(
            dict[str, dict[str, dict[str, str]]],
            self.aggregate_performances(test_performance=test_performance, format_numbers=True),
        )
        # Log average overall and per-perturbation performances
        if "overall" in test_performance_aggregated:
            if "spearman" in test_performance_aggregated["overall"]:
                pf = test_performance_aggregated["overall"]["spearman"]["mean"]
                logger.info(
                    f"Mean Spearman's correlation overall: {pf}",
                )
            if "auc" in test_performance_aggregated["overall"]:
                pf = test_performance_aggregated["overall"]["auc"]["mean"]
                logger.info(f"Mean AUC overall: {pf}")
        if "per_perturbation" in test_performance_aggregated:
            if "spearman" in test_performance_aggregated["per_perturbation"]:
                pf = test_performance_aggregated["per_perturbation"]["spearman"]["mean"]
                logger.info(
                    f"Mean Spearman's correlation per perturbation: {pf}",
                )
            if "auc" in test_performance_aggregated["per_perturbation"]:
                pf = test_performance_aggregated["per_perturbation"]["auc"]["mean"]
                logger.info(f"Mean AUC per perturbation: {pf}")

        return test_performance, test_performance_aggregated

    def compute_performances(
        self,
        test_true_labels: dict[Any, pd.DataFrame],
        test_predicted_labels: dict[Any, pd.DataFrame],
        performance_per_perturbation: bool | list[bool] | None = None,
        metric: RegressionMetricType | list[RegressionMetricType] | None = None,
    ) -> dict[str, dict[str, dict[Any, dict[str, float]]]]:
        """Compute the performance metrics.

        Parameters
        ----------
        test_true_labels : dict[Any, pd.DataFrame]
            Dictionary of true labels in the test set(s).
        test_predicted_labels : dict[Any, pd.DataFrame]
            Dictionary of predicted labels in the test set(s).
        performance_per_perturbation : bool | list[bool] | None
            Whether to evaluate performance per perturbation (True) or over all
            perturbations (False). By default, both per perturbation and overall
            performances are calculated.
        metric : RegressionMetricType | list[RegressionMetricType] | None
            The metric(s) to use for evaluation. Possible values are: "spearman",
            "pearson", "r2", "mse" and "mae". By default, all possible values are
            calculated.

        Returns
        -------
        dict[str, dict[str, dict[Any, dict[str, float]]]]
            The first key can be either "per_perturbation" or "overall". The second key
            is the metric name. The third key is the split id.
            The fourth key is the perturbation name if performance_per_perturbation is True
            and "overall" otherwise. The value is the performance metric.
        """
        # Define arguments
        if performance_per_perturbation is None:
            performance_per_perturbation = [True, False]
        if metric is None:
            metric = ["spearman", "pearson", "r2", "mse", "mae"]

        # Check arguments
        if isinstance(metric, str):
            metric = [metric]
        if isinstance(performance_per_perturbation, bool):
            performance_per_perturbation = [performance_per_perturbation]

        # Initiate the dictionary to store the performance
        test_performance: dict[str, dict[str, dict[Any, dict[str, float]]]] = {}

        # Compute the different performance metrics
        for per_perturbation in performance_per_perturbation:
            performance_type = "per_perturbation" if per_perturbation else "overall"
            test_performance[performance_type] = {}
            for metric_name in metric:
                test_performance[performance_type][metric_name] = {}
                for split, df_predictions in test_predicted_labels.items():
                    # Extract and compare y_true and y_pred
                    if per_perturbation:
                        performance_per_split = {}
                        for label_name in test_true_labels[split].columns:
                            y_predicted_series = df_predictions[label_name]
                            y_true_series = test_true_labels[split][label_name]
                            performance_per_split[label_name] = performance_metric_wrapper(
                                y_true_series,
                                y_predicted_series,
                                metric=metric_name,
                            )
                    else:
                        # Keep perturbations available in the test set
                        y_true_df = test_true_labels[split]
                        y_pred_df = df_predictions
                        common_columns = y_true_df.columns.intersection(y_pred_df.columns)
                        y_true_stacked = y_true_df[common_columns].stack()
                        y_pred_stacked = y_pred_df[common_columns].stack()
                        performance_per_split = {
                            "overall": performance_metric_wrapper(
                                pd.Series(y_true_stacked),
                                pd.Series(y_pred_stacked),
                                metric=metric_name,
                            )
                        }
                    test_performance[performance_type][metric_name][split] = performance_per_split

        return test_performance

    @staticmethod
    def aggregate_performances(
        test_performance: dict[str, dict[str, dict[Any, dict[str, float]]]],
        format_numbers: bool = True,
    ) -> dict[str, dict[str, dict[str, float | str]]]:
        """Aggregate performances over splits and labels.

        Parameters
        ----------
        test_performance : dict[str, dict[str, dict[Any, dict[str, float]]]]
            Dictionary of performances per split, per metric and per perturbation
            (if applicable).
        format_numbers : bool
            Whether to format the numbers to 4 decimal places, by default True.

        Returns
        -------
        dict[str, dict[str, dict[str, float | str]]]
            Dictionary of aggregated performances.
            - First key is the performance type ("per_perturbation" or "overall").
            - Second key is the metric name.
            - Third key is either "mean" or "std" for the overall performance, and "mean_{perturbation}" or
            "std_{perturbation}" for each perturbation if performance_per_perturbation is True.
            - The value is the performance metric.
        """

        def format_value(value: float) -> float | str:
            return f"{value:.4f}" if format_numbers else value

        aggregated: dict[str, dict[str, dict[str, float | str]]] = {}
        for perf_type, metrics in test_performance.items():
            aggregated[perf_type] = {}
            for metric, perturbations in metrics.items():
                df_perf = pd.DataFrame(perturbations)
                aggregated[perf_type][metric] = {
                    "mean": format_value(df_perf.mean().mean()),
                    "std": format_value(df_perf.mean(axis=0).std()),  # std over splits
                }
                if perf_type == "per_perturbation":
                    df = pd.DataFrame(perturbations).T
                    aggregated[perf_type][metric].update(
                        {f"mean_{pert}": format_value(mean) for pert, mean in df.mean().items()}
                    )
                    aggregated[perf_type][metric].update(
                        {f"std_{pert}": format_value(std) for pert, std in df.std().items()}
                    )
        return aggregated

    def save_attribute(self, attribute_name: str, output_path: Path) -> None:
        """Pickle one of the attributes of the current instance of the class.

        Parameters
        ----------
        attribute_name : str
            Name of the attribute to save.
        output_path : Path
            Path of the output file.
        """
        logger.info("Saving the results...")

        # Create the parent directory if it does not exist
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)

        # Pickle the current instance of the selected attribute
        save_pickle(self.__dict__[attribute_name], output_path)
        logger.info(f"{attribute_name} saved at {output_path}")

    def save_trainer(self, output_path: Path) -> None:
        """Pickle the current instance of the class.

        Parameters
        ----------
        output_path : Path
            Path of the output file.
        """
        logger.info("Saving the trainer...")

        # Create the parent directory if it does not exist
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)

        # Pickle the current instance
        save_pickle(self, output_path)
        logger.info(f"Trainer saved at {output_path}")

    def empty_data(self) -> None:
        """Empty the data to save memory."""
        # The rnaseq columns are kept to be able to use predict
        self.data.df_rnaseq = pd.DataFrame(columns=self.data.df_rnaseq.columns)
        for attr in [
            "df_fingerprints",
            "df_labels",
            "df_labels_stacked",
            "df_perturbation_metadata",
            "df_sample_metadata",
            "df_sample_metadata_stacked",
        ]:
            if hasattr(self.data, attr):
                delattr(self.data, attr)

    def convert_to_no_ensembling(self, metric: RegressionMetricType | list[RegressionMetricType] | None = None) -> None:
        """Update the trainer to use trained models without ensembling.

        This method can only be used after the trainer has been run. It computes
        predictions in the test sets using the trained models without ensembling and
        updates the prediction performances.

        Parameters
        ----------
        metric: RegressionMetricType | list[RegressionMetricType] | None
            metric(s) to use for evaluation.
        """
        # Check the ensembling status of the model
        if self.config_model.ensembling is False:
            logger.error("The model is already using no ensembling.")

        # Update the model to use no ensembling
        self.config_model.ensembling = False
        for trained_model in self.trained_model.values():
            trained_model.ensembling = False

        # Load the data if needed
        if not hasattr(self.data, "df_labels"):
            self._data = self.load_data()
        # Update predictions and performances
        self.test_predicted_labels = self.predict_test(data=self.data, split_pair_ids=self.split_pair_ids)
        self.test_performance, self.test_performance_aggregated = self.evaluate(
            test_true_labels=self.get_test_true_labels(data=self.data, split_pair_ids=self.split_pair_ids),
            test_predicted_labels=self.test_predicted_labels,
            metric=metric,
        )

    def predict(
        self,
        X: pd.DataFrame,
        X_metadata: pd.DataFrame | None = None,
        X_fgpt: pd.DataFrame | None = None,
        impute_missing_genes_strategy: str = "zeros",
        refit_preprocessor: bool = True,
        ensemble_over_splits: bool = True,
    ) -> pd.DataFrame | dict[Any, pd.DataFrame]:
        """Predict labels in new data.

        (i) Compute the predictions for all samples and all splits.
        (ii) Compute the average over all the models' predictions to return a single prediction per sample.

        Parameters
        ----------
        X : pd.DataFrame
            The new data to predict on.
        X_metadata : pd.DataFrame | None
            Metadata for the samples in X, by default None.
        X_fgpt : pd.DataFrame | None
            The fingerprints of the new data, by default None.
        impute_missing_genes_strategy : str
            The strategy to impute the genes which are given in X but were missing at the training in a given model,
            possible options: 'zeros', by default 'zeros'.
        refit_preprocessor : bool
            Whether to refit the preprocessor on the input data, by default True. This is recommended if the input data
            is from a different study than the training data. Re-fitting the preprocessor can be seen as a simple data
            alignment procedure.
        ensemble_over_splits : bool
            Whether to ensemble over the splits, by default True.

        Returns
        -------
        pd.DataFrame | dict[Any, pd.DataFrame]
            The predicted labels.
            If ensemble_over_splits is True, the average of the predictions over all
            splits is returned. Otherwise, the predictions for each split are returned,
            as a dictionary with the split id as key.

        Raises
        ------
        ValueError
            If the strategy to impute missing genes is not supported.
        """
        # Compute the predictions for all samples and all splits
        predictions = {}

        # Create dummy X_metadata if needed
        if X_metadata is None:
            X_metadata = pd.DataFrame(index=X.index)

        # Ensure that X and X_metadata ids are in the same order
        X_columns = X.columns
        X = X.loc[X_metadata.index, X_columns.isin(self.data.df_rnaseq.columns)]
        X_metadata = X_metadata.loc[X.index]

        # Impute missing genes if needed
        rnaseq_columns = set(self.data.df_rnaseq.columns)
        missing_genes = rnaseq_columns - set(X_columns)
        if len(missing_genes) > 0:
            # Check that there are common genes between the input and the model
            if len(rnaseq_columns.intersection(X_columns)) == 0:
                raise ValueError("No common genes between the input and the model.")

            # Report the number of missing genes
            logger.info(
                f"Imputing {len(missing_genes)} of the {len(rnaseq_columns)} required genes that are not available in "
                "the input data."
            )

            missing_genes_list = list(missing_genes)
            if impute_missing_genes_strategy == "zeros":
                X = pd.concat([X, pd.DataFrame(0, index=X.index, columns=missing_genes_list)], axis=1)
            else:
                raise ValueError(
                    f"Strategy {impute_missing_genes_strategy} is not supported for imputing missing genes."
                )

        # iterate through every model
        for split in self.split_pair_ids:
            if refit_preprocessor:
                # Refit the preprocessor that was used during training
                preprocessor = copy.deepcopy(self.trained_model[split].trained_preprocessor)
                # Remove _rnaseq suffix for preprocessing
                X_no_suffix = X.copy()
                X_no_suffix.columns = X_no_suffix.columns.str.replace("_rnaseq", "", regex=False)
                if preprocessor is not None:
                    preprocessor.fit(X_no_suffix)
                    X_preprocessed = preprocessor.transform(X_no_suffix)
                else:
                    X_preprocessed = X_no_suffix
                # Add _rnaseq suffix back
                X_preprocessed = X_preprocessed.copy()
                X_preprocessed.columns = X_preprocessed.columns + "_rnaseq"
            else:
                X_preprocessed = X

            # Fingerprints are always loaded in the trainer but only used by some models
            predictions[split] = self.trained_model[split].predict(
                X=X_preprocessed,
                X_fgpt=X_fgpt,
                X_metadata=X_metadata,
                preprocessor_transform=not refit_preprocessor,
            )

        if ensemble_over_splits:
            # Average all the values from the keys in the dictionary
            summed_predictions = sum(predictions.values())
            predictions_df = summed_predictions / len(predictions)
            return predictions_df

        # Return the predictions for each split
        return predictions
