"""Perturbation pipeline for LEAP. Includes pre-processing, representation learning and regression."""

import copy
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import ray
from loguru import logger
from tqdm import tqdm

from leap.data.preprocessor import OmicsPreprocessor
from leap.metrics.regression_metrics import RegressionMetricType, performance_metric_wrapper
from leap.regression_models import ElasticNet, KnnRegressor, RegressionModel
from leap.representation_models import RepresentationModelBase
from leap.utils.io import load_pickle, save_pickle


# Constants for magic strings
FOLD_PREFIX = "fold_"
FULL_TRAINING_KEY = "full_training_data"
SAMPLE_INDEX = "sample"
PERTURBATION_INDEX = "perturbation"


class PerturbationPipeline:
    """Class for the perturbation regression model.

    Parameters
    ----------
    preprocessor_model_rnaseq : OmicsPreprocessor | Path | None
        Path to a trained preprocessor model or configuration of a preprocessor model to preprocess the input RNASeq
        data. If None, no preprocessing is applied.
    rpz_model_rnaseq : RepresentationModelBase | Path | None
        Path to a trained RPZ model or configuration of a RPZ model to train the RPZ for RNASeq when running .fit().
        If None, no representation learning is applied.
    regression_model_base_instance : RegressionModel
        Configuration of the prediction model.
    hpt_tuning_cv_split : Callable | None
        Configuration to define the folds used in the cross validation for hyper-parameter tuning. The models trained in
        each of these splits are ensembled if ensembling is True.
    hpt_tuning_param_grid : dict | None
        Configuration to define the grid of hyper-paramaters to try in the cross validation for hyper-parameter tuning.
    hpt_tuning_score : RegressionMetricType | None
        Score or scoring function to maximise for hyper-parameter tuning. Possible values are: "spearman", "pearson",
        "r2", "mse", "mae".
    fgpt_rpz_model : RepresentationModelBase | Path | None
        Path to a trained RPZ model for fingerprints or a RPZ model for fingerprints to train the RPZ when running
        .fit(). If None, no representation learning is applied to fingerprints.
    one_model_per_perturbation : bool
        Whether to train one model per perturbation, by default True.
    ensembling : bool
        Whether to average the predictions of the models trained in the different CV folds, by default True. Note that
        ensembling cannot be set to True if hpt_tuning_cv_split is not provided. Default is True.
    ensembling_save_models_to_disk : bool
        Whether to save the ensembling models to disk, this can save RAM and prevent out of memory errors for heavy
        models, by default False.
    use_ray : bool
        Whether to use Ray to parallelise over the perturbations. Only possible if one_model_per_perturbation is True.
        Default is False.
    ray_remote_params : dict | None
        Parameters for Ray remote. Defaults to None.

    Raises
    ------
    ValueError
        If hpt_tuning_score is not provided when hpt_tuning_param_grid is provided.
    """

    def __init__(
        self,
        preprocessor_model_rnaseq: OmicsPreprocessor | Path | None,
        rpz_model_rnaseq: RepresentationModelBase | Path | None,
        regression_model_base_instance: RegressionModel,
        hpt_tuning_cv_split: Callable | None,
        hpt_tuning_param_grid: dict | None,
        hpt_tuning_score: RegressionMetricType | None,
        fgpt_rpz_model: RepresentationModelBase | Path | None,
        one_model_per_perturbation: bool = True,
        ensembling: bool = True,
        ensembling_save_models_to_disk: bool = False,
        use_ray: bool = False,
        ray_remote_params: dict | None = None,
    ):
        # Store arguments
        self.preprocessor_model_rnaseq = preprocessor_model_rnaseq
        self.rpz_model_rnaseq = rpz_model_rnaseq
        self.fgpt_rpz_model = fgpt_rpz_model
        self.regression_model_base_instance = regression_model_base_instance
        self.hpt_tuning_cv_split = hpt_tuning_cv_split
        self.hpt_tuning_param_grid = hpt_tuning_param_grid
        self.hpt_tuning_score = hpt_tuning_score
        self.one_model_per_perturbation = one_model_per_perturbation
        self.ensembling = ensembling
        self.ensembling_save_models_to_disk = ensembling_save_models_to_disk
        self.use_ray = use_ray
        self.ray_remote_params = ray_remote_params or {"num_cpus": 1}
        self.ensembling_output_path: Path | None = None

        # Check that arguments are compatible
        if self.use_ray and not self.one_model_per_perturbation:
            logger.warning("Ray is only available for one model per perturbation.")
            self.use_ray = False
        if self.hpt_tuning_cv_split is None:
            if self.ensembling:
                logger.warning("ensembling is set to False as hpt_tuning_cv_split is None.")
                self.ensembling = False
            if self.hpt_tuning_param_grid is not None:
                logger.warning("hpt_tuning_param_grid is ignored as hpt_tuning_cv_split is None.")
        elif self.hpt_tuning_param_grid is None:
            self.hpt_tuning_param_grid = self.regression_model_base_instance.get_params()
            self.hpt_tuning_param_grid = {k: [v] for k, v in self.hpt_tuning_param_grid.items()}
        if self.hpt_tuning_cv_split and self.hpt_tuning_score is None:
            raise ValueError("hpt_tuning_score must be provided if hpt_tuning_cv_split is provided.")

        # Initialise trained models (simplified from dict structure since only handling rnaseq)
        self.trained_preprocessor: OmicsPreprocessor | None = None
        self.trained_rpz_model: RepresentationModelBase | None = None
        self.trained_fgpt_rpz_model: RepresentationModelBase | None = None

        # Nested dict for per-perturbation models, simple dict for pan-perturbation
        self.trained_regression_model: dict[str, dict[str, RegressionModel]] | dict[str, RegressionModel] = {}
        self.grid_search_regression_model: dict[str, pd.DataFrame] = {}
        self.y_columns: pd.Index | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_metadata: pd.DataFrame | None = None,
        X_fgpt: pd.DataFrame | None = None,
    ) -> None:
        """Fit the perturbation prediction pipeline.

        This function fits a prediction model for each label in y.
        There are four steps:
        1- Preprocess X using a (trained) preprocessor.
        2- Transform X using a (trained) RPZ model.
        3- Transform X_fgpt using a (trained) RPZ model. Optional, this is only done if X_fgpt is provided.
        4- Train a prediction model for each label in y.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.DataFrame
            Labels.
        X_metadata : pd.DataFrame | None
            Metadata including the grouping variable for preprocessor and/or the grouping variable for cross-validation
            splits as columns. Default is None.
        X_fgpt : pd.DataFrame | None
            Fingerprints. Default is None.

        Raises
        ------
        ValueError
            If fingerprint data is not provided when fitting and one_model_per_perturbation is False.
        """
        # Ensure metadata exists
        X_metadata = self._ensure_metadata(X, X_metadata)

        # Preprocess the input data
        X = self._preprocessor_transform(X, allow_fit=True)

        # Transform input data using rpz model
        X = self._rpz_transform(X, allow_fit=True)

        # Fit regression models
        if self.one_model_per_perturbation:
            # Fit one model per perturbation (fingerprint data is not used)
            self._fit_per_perturbation(X=X, y=y, X_metadata=X_metadata)
        else:
            if X_fgpt is None:
                raise ValueError(
                    "Fingerprint data must be provided when fitting and one_model_per_perturbation is False."
                )
            # Transform fingerprint data using fingerprint rpz model
            X_fgpt = self._rpz_fgpt_transform(X_fgpt, allow_fit=True)
            # Fit one model for all perturbations
            self._fit_all_perturbations(X=X, y=y, X_fgpt=X_fgpt, X_metadata=X_metadata)

    def _ensure_metadata(self, X: pd.DataFrame, X_metadata: pd.DataFrame | None) -> pd.DataFrame:
        """Create empty metadata dataframe if None."""
        if X_metadata is None:
            return pd.DataFrame(index=X.index)
        return X_metadata

    def _preprocessor_transform(self, X: pd.DataFrame, allow_fit: bool = True) -> pd.DataFrame:
        """Transform the input data using the preprocessor model.

        If preprocessor_model is a preprocessor model and if allow_fit is True, this preprocessor model is trained on
        the input data X and used to transform it. The trained preprocessor model is then stored as an attribute of the
        class called trained_preprocessor. The argument allow_fit is True when calling this function in the fit method
        on source domain data only, i.e. training is not permitted when using predict.

        If preprocessor_model is a path to a trained preprocessor model, this preprocessor model is directly used to
        transform the input data X. It is also stored in trained_preprocessor.

        If preprocessor_model is None, the input data is not transformed.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to preprocess.
        allow_fit : bool
            Whether to allow fitting the preprocessor. Default is True.

        Returns
        -------
        pd.DataFrame
            Preprocessed data.

        Raises
        ------
        RuntimeError
            If the preprocessor model is not fitted before transformation.
        TypeError
            If the preprocessor model is not a OmicsPreprocessor.
        """
        if self.preprocessor_model_rnaseq is None:
            return X

        if allow_fit:
            if isinstance(self.preprocessor_model_rnaseq, Path):
                # Load the trained preprocessor
                self.trained_preprocessor = load_pickle(self.preprocessor_model_rnaseq)
            else:
                # Check the type of preprocessor
                if not isinstance(self.preprocessor_model_rnaseq, OmicsPreprocessor):
                    raise TypeError("The preprocessor needs to be a OmicsPreprocessor.")
                # Train the preprocessor on the data
                self.preprocessor_model_rnaseq.fit(X)
                self.trained_preprocessor = self.preprocessor_model_rnaseq

        if self.trained_preprocessor is None:
            raise RuntimeError("Preprocessor must be fitted before transformation.")

        return self.trained_preprocessor.transform(X)

    def _rpz_transform(self, X: pd.DataFrame, allow_fit: bool = True) -> pd.DataFrame:
        """Transform the input data using the RPZ model.

        If rpz_model is a RPZ model and if allow_fit is True, this model is trained on the input data X and used to
        transform it. The trained RPZ model is then stored as an attribute of the class called trained_rpz_model.

        If rpz_model is a path to a trained RPZ model, this model is directly used to transform the input data X.
        It is also stored in trained_rpz_model.

        If rpz_model is None, the input data is not transformed.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
        allow_fit : bool
            Whether to allow fitting the RPZ model. Default is True.

        Returns
        -------
        pd.DataFrame
            Transformed data.

        Raises
        ------
        RuntimeError
            If the RPZ model is not fitted before transformation.
        """
        if self.rpz_model_rnaseq is None:
            return X

        if allow_fit:
            if isinstance(self.rpz_model_rnaseq, Path):
                # Load the trained RPZ model
                self.trained_rpz_model = load_pickle(self.rpz_model_rnaseq)
            else:
                # Train the RPZ on the data
                self.rpz_model_rnaseq.fit(X)
                self.trained_rpz_model = self.rpz_model_rnaseq

        if self.trained_rpz_model is None:
            raise RuntimeError("RPZ model must be fitted before transformation.")

        X_transformed = self.trained_rpz_model.transform(X)
        return pd.DataFrame(
            X_transformed,
            index=X.index,
            columns=[f"rnaseq_rpz_{i}" for i in range(self.trained_rpz_model.repr_dim)],
        )

    def _rpz_fgpt_transform(self, X_fgpt: pd.DataFrame, allow_fit: bool = True) -> pd.DataFrame:
        """Transform fingerprint data using the fingerprint RPZ model.

        Parameters
        ----------
        X_fgpt : pd.DataFrame
            Fingerprint data to transform.
        allow_fit : bool
            Whether to allow fitting the model. Default is True.

        Returns
        -------
        pd.DataFrame
            Transformed fingerprint data.

        Raises
        ------
        RuntimeError
            If the fingerprint RPZ model is not fitted before transformation.
        """
        if self.fgpt_rpz_model is None:
            return X_fgpt

        if allow_fit:
            if isinstance(self.fgpt_rpz_model, Path):
                # Load the trained fingerprint RPZ model
                self.trained_fgpt_rpz_model = load_pickle(self.fgpt_rpz_model)
            else:
                # Train the fingerprint RPZ on the data
                self.fgpt_rpz_model.fit(X_fgpt)
                self.trained_fgpt_rpz_model = self.fgpt_rpz_model

        if self.trained_fgpt_rpz_model is None:
            raise RuntimeError("Fingerprint RPZ model must be fitted before transformation.")

        X_fgpt_transformed = self.trained_fgpt_rpz_model.transform(X_fgpt)
        return pd.DataFrame(
            X_fgpt_transformed,
            index=X_fgpt.index,
            columns=[f"fgpt_rpz_{i}" for i in range(self.trained_fgpt_rpz_model.repr_dim)],
        )

    def _fit_per_perturbation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_metadata: pd.DataFrame,
    ) -> None:
        """Fit one model per perturbation."""
        # Define index names
        X.index.name = SAMPLE_INDEX
        y.index.name = SAMPLE_INDEX
        X_metadata.index.name = SAMPLE_INDEX

        # Initialize as nested dict for per-perturbation models
        self.trained_regression_model = {}

        # Fit models for each perturbation
        if self.use_ray:
            dict_grid_per_perturbation = self._fit_with_ray(X, y, X_metadata)
        else:
            dict_grid_per_perturbation = self._fit_sequential(X, y, X_metadata)

        # Store the outputs in attributes
        self._store_per_perturbation_results(dict_grid_per_perturbation)

    def _fit_with_ray(self, X: pd.DataFrame, y: pd.DataFrame, X_metadata: pd.DataFrame) -> dict:
        """Fit models using Ray for parallelization."""

        @ray.remote(**self.ray_remote_params)
        def _fit_one_perturbation_with_ray(*args: Any, **kwargs: Any) -> dict[str, Any]:
            """Wrap _fit_one_perturbation to flag it with ray.remote()."""
            return _fit_one_perturbation(*args, **kwargs)

        ray.init(ignore_reinit_error=True)
        futures = {
            label: _fit_one_perturbation_with_ray.remote(
                x_data=X,
                y_data=y[label],
                X_metadata=X_metadata,
                regression_model_base_instance=self.regression_model_base_instance,
                hpt_tuning_param_grid=self.hpt_tuning_param_grid,
                hpt_tuning_cv_split=self.hpt_tuning_cv_split,
                hpt_tuning_score=self.hpt_tuning_score,
                ensembling=self.ensembling,
                ensembling_save_models_to_disk=self.ensembling_save_models_to_disk,
                ensembling_output_path=self.ensembling_output_path,
                pbar=None,
            )
            for label in y.columns
        }
        return {key: ray.get(value) for key, value in futures.items()}

    def _fit_sequential(self, X: pd.DataFrame, y: pd.DataFrame, X_metadata: pd.DataFrame) -> dict:
        """Fit models sequentially with progress bar."""
        pbar = tqdm(y.columns, desc="Perturbation", leave=True)
        return {
            label: _fit_one_perturbation(
                x_data=X,
                y_data=y[label],
                X_metadata=X_metadata,
                regression_model_base_instance=self.regression_model_base_instance,
                hpt_tuning_param_grid=self.hpt_tuning_param_grid,
                hpt_tuning_cv_split=self.hpt_tuning_cv_split,
                hpt_tuning_score=self.hpt_tuning_score,
                ensembling=self.ensembling,
                ensembling_save_models_to_disk=self.ensembling_save_models_to_disk,
                ensembling_output_path=self.ensembling_output_path,
                pbar=pbar,
            )
            for label in pbar
        }

    def _store_per_perturbation_results(self, dict_grid_per_perturbation: dict) -> None:
        """Store trained models and grid search results."""
        # Type narrow: we know this is nested dict for per-perturbation
        trained_models = cast(dict[str, dict[str, RegressionModel]], self.trained_regression_model)

        for label, grid in dict_grid_per_perturbation.items():
            if self.ensembling:
                for key, model in grid["best_ensemble_models_"].items():
                    if key not in trained_models:
                        trained_models[key] = {}
                    trained_models[key][label] = model

            # Save trained regression model on all training data
            if FULL_TRAINING_KEY not in trained_models:
                trained_models[FULL_TRAINING_KEY] = {}
            trained_models[FULL_TRAINING_KEY][label] = grid["best_estimator_"]

            # Save grid search results (None if no HPT)
            self.grid_search_regression_model[label] = grid["cv_results_"]

    def _fit_all_perturbations(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_fgpt: pd.DataFrame,
        X_metadata: pd.DataFrame,
    ) -> None:
        """Fit one model for all perturbations."""
        logger.info("Training a single regression model on all perturbations...")

        # Initialize as simple dict for pan-perturbation model
        self.trained_regression_model = {}

        # Melt data if not using KNN (KNN handles multilabel directly)
        if isinstance(self.regression_model_base_instance, KnnRegressor):
            # Use common perturbations
            common_perturbations = list(y.columns.intersection(X_fgpt.index))
            if len(common_perturbations) == 0:
                raise ValueError("No common perturbations found between y and X_fgpt.")
            if len(y.columns) != len(common_perturbations):
                logger.warning(f"Perturbations not in fingerprints: {set(y.columns) - set(common_perturbations)}")
                logger.warning(f"Perturbations not in labels: {set(X_fgpt.index) - set(common_perturbations)}")
            y = y[common_perturbations]
            X_fgpt = X_fgpt.loc[pd.Index(common_perturbations)]
            X, X_metadata, y = self._melt_data(X, X_metadata, X_fgpt, y)

        # Fit the model on all perturbations
        grid = _fit_single_label(
            x_data=X,
            y_data=y,
            X_metadata=X_metadata,
            regression_model_base_instance=self.regression_model_base_instance,
            hpt_tuning_param_grid=self.hpt_tuning_param_grid,
            hpt_tuning_cv_split=self.hpt_tuning_cv_split,
            hpt_tuning_score=self.hpt_tuning_score,
            ensembling=self.ensembling,
            ensembling_save_models_to_disk=self.ensembling_save_models_to_disk,
            ensembling_output_path=self.ensembling_output_path,
        )

        # Type narrow: we know this is simple dict for pan-perturbation
        trained_models = cast(dict[str, RegressionModel], self.trained_regression_model)

        if self.ensembling:
            for key, model in grid["best_ensemble_models_"].items():
                trained_models[key] = model

        # Save trained regression model on all training data
        trained_models[FULL_TRAINING_KEY] = grid["best_estimator_"]

        # Save grid search results
        self.grid_search_regression_model = grid["cv_results_"]

        # Store column names for prediction
        self.y_columns = y.columns

    def _melt_data(
        self,
        X: pd.DataFrame,
        X_metadata: pd.DataFrame,
        X_fgpt: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Melt the data for pan-perturbation models.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        X_metadata : pd.DataFrame
            Metadata.
        X_fgpt : pd.DataFrame
            Fingerprints.
        y : pd.DataFrame | None
            Labels, by default None.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Melted input data, melted metadata and melted labels.
        """
        if y is None:
            # Create dummy label data
            y = pd.DataFrame(index=X.index, columns=X_fgpt.index)

        # Melt the label data
        y.index.name = SAMPLE_INDEX
        y.columns.name = PERTURBATION_INDEX
        y_concat = y.reset_index().melt(id_vars=SAMPLE_INDEX)

        # Align the input data
        X_concat = y_concat.join(X, on=SAMPLE_INDEX).drop("value", axis=1)
        X_fgpt.index.name = PERTURBATION_INDEX
        X_fgpt_concat = y_concat.join(X_fgpt, on=PERTURBATION_INDEX).drop("value", axis=1)
        X_concat = X_concat.set_index([PERTURBATION_INDEX, SAMPLE_INDEX]).sort_index()
        X_fgpt_concat = X_fgpt_concat.set_index([PERTURBATION_INDEX, SAMPLE_INDEX]).sort_index()
        y_concat = y_concat.set_index([PERTURBATION_INDEX, SAMPLE_INDEX]).sort_index()
        X_full_concat = pd.concat([X_concat, X_fgpt_concat], axis=1)

        # Align the metadata with y index
        X_metadata_concat = y_concat.merge(X_metadata, left_on=SAMPLE_INDEX, right_index=True)[X_metadata.columns]
        X_metadata_concat[SAMPLE_INDEX] = X_metadata_concat.index.get_level_values(SAMPLE_INDEX)
        X_metadata_concat[PERTURBATION_INDEX] = X_metadata_concat.index.get_level_values(PERTURBATION_INDEX)

        return X_full_concat, X_metadata_concat, y_concat

    def _load_model_from_path(self, model_path: Path) -> Any:
        """Load a model from disk."""
        if model_path.suffix == ".pkl":
            return load_pickle(model_path)
        raise NotImplementedError(f"Loading model under type {model_path.suffix} is not implemented for ensembling")

    def _get_average_fold_prediction_for_ensembling(
        self, X: pd.DataFrame, label_name: str | None = None, columns: list[str] | None = None
    ) -> pd.Series | pd.DataFrame:
        """Average predictions across all fold models for ensembling.

        Parameters
        ----------
        X : pd.DataFrame
            Input data for prediction.
        label_name : str | None
            Label name for per-perturbation models. None for pan-perturbation models.
        columns : list[str] | None
            Column names for DataFrame output. If None, returns Series.

        Returns
        -------
        pd.Series | pd.DataFrame
            Averaged predictions.

        Raises
        ------
        TypeError
            If model_or_dict is not a dict when label_name is provided.
        """
        models_results = []
        for key, model_or_dict in self.trained_regression_model.items():
            if FOLD_PREFIX in key:
                if label_name is not None:
                    # Per-perturbation case
                    if not isinstance(model_or_dict, dict):
                        raise TypeError("model_or_dict should be a dict when label_name is provided")
                    model_path = model_or_dict[label_name]
                else:
                    # Pan-perturbation case - type narrow since label_name is None
                    if isinstance(model_or_dict, dict):
                        raise TypeError("model_or_dict should not be a dict for pan-perturbation models")
                    model_path = model_or_dict

                # Check if model_path is a path or a model
                if isinstance(model_path, Path):
                    loaded_model = self._load_model_from_path(model_path)
                else:
                    loaded_model = model_path
                models_results.append(loaded_model.predict(X))

        avg_predictions = np.mean(models_results, axis=0)

        if columns is None:
            return pd.Series(avg_predictions, index=X.index)
        return pd.DataFrame(avg_predictions, index=X.index, columns=columns)

    def predict(
        self,
        X: pd.DataFrame,
        X_metadata: pd.DataFrame | None = None,
        X_fgpt: pd.DataFrame | None = None,
        list_of_perturbations: list | None = None,
        preprocessor_transform: bool = True,
    ) -> pd.DataFrame:
        """Predict the labels for X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        X_metadata : pd.DataFrame | None
            Metadata including the grouping variable for preprocessor and/or the grouping variable for cross-validation
            splits as columns. Default is None.
        X_fgpt : pd.DataFrame | None
            Fingerprints.
        list_of_perturbations : list | None
            List of perturbations to predict labels for, by default None.
        preprocessor_transform : bool
            Whether to transform the input data using the preprocessor, by default True.

        Returns
        -------
        pd.DataFrame
            Predicted labels.
        """
        # Ensure metadata exists
        X_metadata = self._ensure_metadata(X, X_metadata)

        # Preprocess the input data
        if preprocessor_transform:
            X = self._preprocessor_transform(X, allow_fit=False)

        # Transform input data using rpz models
        X = self._rpz_transform(X, allow_fit=False)

        # Predict based on model type
        if self.one_model_per_perturbation:
            return self._predict_per_perturbation(X, X_fgpt, list_of_perturbations)
        elif not isinstance(self.regression_model_base_instance, KnnRegressor):
            return self._predict_single_model_with_fgpt(X, X_metadata, X_fgpt, list_of_perturbations)
        else:
            return self._predict_multilabel(X, list_of_perturbations)

    def _predict_per_perturbation(
        self, X: pd.DataFrame, X_fgpt: pd.DataFrame | None, list_of_perturbations: list | None
    ) -> pd.DataFrame:
        """Predict using per-perturbation models."""
        # Get trained models
        fold_id = FULL_TRAINING_KEY if FULL_TRAINING_KEY in self.trained_regression_model else f"{FOLD_PREFIX}0"
        trained_regression_model_fold = self.trained_regression_model[fold_id]
        if not isinstance(trained_regression_model_fold, dict):
            raise TypeError("trained_regression_model_fold should be a dictionary for per-perturbation models.")

        seen_perturbations = list(trained_regression_model_fold.keys())

        # Define the list of all perturbations to predict
        if list_of_perturbations is None:
            list_of_perturbations = X_fgpt.index.tolist() if X_fgpt is not None else seen_perturbations

        if len(list_of_perturbations) == 0:
            raise ValueError("No perturbations to predict.")

        # Check for unseen perturbations
        unseen_perturbations = set(list_of_perturbations) - set(seen_perturbations)
        if unseen_perturbations:
            raise NotImplementedError(
                f"Predicting unseen perturbations is not supported. Unseen: {unseen_perturbations}"
            )

        # Predict for each perturbation
        data = {}
        for label_name in list_of_perturbations:
            if self.ensembling:
                # Average over models trained in the different CV folds
                data[label_name] = self._get_average_fold_prediction_for_ensembling(X, label_name)
            else:
                models_full_training_data = self.trained_regression_model[FULL_TRAINING_KEY]
                if not isinstance(models_full_training_data, dict):
                    raise TypeError("models_full_training_data should be a dict for per-perturbation models.")
                model_label = models_full_training_data[label_name]
                if model_label is None:
                    raise RuntimeError(f"Model for {label_name} must be trained before prediction.")
                data[label_name] = pd.Series(model_label.predict(X), index=X.index)

        # Concatenate all predictions in a dataframe
        return pd.concat(data, axis=1)

    def _predict_single_model_with_fgpt(
        self,
        X: pd.DataFrame,
        X_metadata: pd.DataFrame,
        X_fgpt: pd.DataFrame | None,
        list_of_perturbations: list | None,
    ) -> pd.DataFrame:
        """Predict using single model with fingerprints."""
        if X_fgpt is None:
            raise ValueError("Fingerprint data must be provided for a single model on all perturbations.")

        # Transform the fingerprint data using trained rpz
        if self.trained_fgpt_rpz_model is None:
            raise RuntimeError("Fingerprint RPZ model must be trained before prediction.")

        X_fgpt_transformed = pd.DataFrame(
            self.trained_fgpt_rpz_model.transform(X_fgpt),
            index=X_fgpt.index,
            columns=[f"fgpt_rpz_{i}" for i in range(self.trained_fgpt_rpz_model.repr_dim)],
        )

        # Keep the perturbations that we want to predict only
        if list_of_perturbations is not None:
            if len(list_of_perturbations) == 0:
                raise ValueError("list_of_perturbations is empty.")
            X_fgpt_transformed = X_fgpt_transformed.loc[list_of_perturbations]

        # Melt the data
        X_melted, _, _ = self._melt_data(X, X_metadata, X_fgpt_transformed)

        # Predict
        if self.ensembling:
            df_predicted = self._get_average_fold_prediction_for_ensembling(X_melted, columns=["predicted"])
        else:
            model_full_data = self.trained_regression_model[FULL_TRAINING_KEY]
            if isinstance(model_full_data, dict):
                raise TypeError(
                    "trained_regression_model['full_training_data'] should not be a dictionary "
                    "for pan-perturbation models."
                )
            if model_full_data is None:
                raise RuntimeError("Model must be trained before prediction.")
            df_predicted = pd.DataFrame(model_full_data.predict(X_melted), index=X_melted.index, columns=["predicted"])

        # Pivot the dataframe
        return df_predicted.reset_index().pivot(index=SAMPLE_INDEX, columns=PERTURBATION_INDEX, values="predicted")

    def _predict_multilabel(self, X: pd.DataFrame, list_of_perturbations: list | None) -> pd.DataFrame:
        """Predict using multilabel regressor (no fingerprints)."""
        # Type narrow: we know this is simple dict for pan-perturbation
        trained_regression_model = cast(dict[str, RegressionModel], self.trained_regression_model)

        if self.y_columns is None:
            raise RuntimeError("Labels must be provided for multilabel regressor.")

        if list_of_perturbations is not None:
            unseen = [p for p in list_of_perturbations if p not in self.y_columns]
            if unseen:
                raise ValueError(f"The multilabel regressor cannot predict unseen perturbations: {unseen}")

        # Predict
        if self.ensembling:
            predictions = [model.predict(X) for model in trained_regression_model.values()]
            df_predicted = pd.DataFrame(np.mean(predictions, axis=0), index=X.index, columns=self.y_columns)
        else:
            model_full_data = trained_regression_model[FULL_TRAINING_KEY]
            df_predicted = pd.DataFrame(model_full_data.predict(X), index=X.index, columns=self.y_columns)

        return df_predicted


def _add_perturbation_index(df: pd.DataFrame | pd.Series, perturbation_name: str) -> pd.DataFrame | pd.Series:
    """Add the perturbation name to a DataFrame in a multi index."""
    df.index = pd.MultiIndex.from_arrays(
        [np.full(len(df), perturbation_name), df.index.get_level_values(SAMPLE_INDEX)],
        names=[PERTURBATION_INDEX, SAMPLE_INDEX],
    )
    return df


def _drop_missing_values_if_any(
    x_data: pd.DataFrame,
    y_data: pd.Series | pd.DataFrame,
    X_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame, pd.DataFrame]:
    """Drop rows with missing values in y_data (for single-label case)."""
    if isinstance(y_data, pd.Series) or y_data.shape[1] == 1:
        y_data = y_data.dropna()
        x_data = x_data.loc[y_data.index]
        X_metadata = X_metadata.loc[y_data.index]
    return x_data, y_data, X_metadata


def _define_param_grid(
    x_data: pd.DataFrame,
    y_data: pd.Series | pd.DataFrame,
    hpt_tuning_param_grid: dict,
    l1_ratio: float | None = None,
) -> dict:
    """Define the grids of hyper-parameters."""
    param_grid_values = {}
    for param_name, hpt_tuning_param_value in hpt_tuning_param_grid.items():
        # Check if the argument has a get_alpha_grid method (for AlphaGridElasticNet)
        if hasattr(hpt_tuning_param_value, "get_alpha_grid"):
            # This is currently only available for the elastic net
            param_grid_values[param_name] = hpt_tuning_param_value.get_alpha_grid(X=x_data, y=y_data, l1_ratio=l1_ratio)
        else:
            # Store the list of parameter values to visit in grid search
            param_grid_values[param_name] = hpt_tuning_param_value
    return param_grid_values


def _param_search(
    param_grid: dict,
    regression_model_base_instance: RegressionModel,
    x_data: pd.DataFrame,
    y_data: pd.DataFrame | pd.Series,
    X_metadata: pd.DataFrame,
    hpt_tuning_cv_split: Callable,
    hpt_tuning_score: RegressionMetricType | None,
    param_search_type: dict | None = None,
    ensembling: bool = True,
    ensembling_save_models_to_disk: bool = False,
    ensembling_output_path: Path | None = None,
) -> dict:
    """Perform grid search with cross-validation."""
    if hpt_tuning_score is None:
        raise ValueError("A score has to be provided for the grid search.")

    models_params = define_model_params(param_grid=param_grid, param_search_type=param_search_type)

    best_score = -np.inf
    best_model = None
    best_ensemble_models = {}
    all_results = []

    # Iterate over all combinations of hyperparameters
    for current_params in models_params:
        fold_scores = []
        fold_models = {}

        # Manually iterate over the cross-validation splits
        hpt_cv_splits = hpt_tuning_cv_split(X_metadata=X_metadata)
        for i, (train_index, val_index) in enumerate(hpt_cv_splits):
            X_train, X_val = x_data.iloc[train_index], x_data.iloc[val_index]
            y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

            # Copy the model, set params and train
            model = copy.deepcopy(regression_model_base_instance)
            model.set_params(**current_params)
            model.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)

            # Predict on the validation fold
            y_pred = pd.Series(model.predict(X_val), index=X_val.index)

            # Reformat y_val as a series if it is a dataframe with one column
            if isinstance(y_data, pd.DataFrame) and y_val.shape[1] == 1:
                y_val = y_val.iloc[:, 0]

            # Calculate the score per perturbation
            score = performance_metric_wrapper(
                y_true=y_val, y_pred=y_pred, metric=hpt_tuning_score, per_perturbation=True
            )

            # Handle greater is better or lower is better metrics
            if hpt_tuning_score in {"mse", "mae"}:
                score = -score

            fold_scores.append(score)
            if ensembling:
                fold_models[f"{FOLD_PREFIX}{i}"] = model

        # Compute average score across folds
        mean_cv_score = float(np.mean(fold_scores))

        # Store results
        all_results.append({"params": current_params, "mean_test_score": mean_cv_score, "cv_scores": fold_scores})

        # Update the best score and model if the current score is better
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = copy.deepcopy(regression_model_base_instance)
            best_model.set_params(**current_params)
            if ensembling:
                best_ensemble_models = fold_models

    # Refit the best model on the full dataset with the best parameters
    if best_model is None:
        raise RuntimeError("No best model found. The grid search was empty.")

    best_model.fit(x_data, y_data)

    # Save all the models for each fold to the disk
    best_ensemble_models_paths = {}
    if ensembling and ensembling_save_models_to_disk and ensembling_output_path is not None:
        ensembling_output_path.mkdir(parents=True, exist_ok=True)
        for key, model in best_ensemble_models.items():
            perturbation_name = X_metadata[PERTURBATION_INDEX].iloc[0].replace("/", "")
            model_path = ensembling_output_path / f"{perturbation_name}_model_{key}.pkl"
            save_pickle(model, model_path)
            best_ensemble_models_paths[key] = model_path
        del best_ensemble_models
        return {
            "best_estimator_": best_model,
            "cv_results_": all_results,
            "best_ensemble_models_": best_ensemble_models_paths,
        }

    return {
        "best_estimator_": best_model,
        "cv_results_": all_results,
        "best_ensemble_models_": best_ensemble_models,
    }


def _fit_single_label(
    x_data: pd.DataFrame,
    y_data: pd.Series | pd.DataFrame,
    X_metadata: pd.DataFrame,
    regression_model_base_instance: RegressionModel,
    hpt_tuning_param_grid: dict | None,
    hpt_tuning_cv_split: Callable | None,
    hpt_tuning_score: RegressionMetricType | None,
    param_search_type: dict | None = None,
    ensembling: bool = True,
    ensembling_save_models_to_disk: bool = False,
    ensembling_output_path: Path | None = None,
) -> dict | dict[str, Any]:
    """Fit the model on a single label.

    Note: this function is also used for the multi-label approach where the y_data
    is a DataFrame with multiple columns.
    """
    if isinstance(regression_model_base_instance, KnnRegressor) and not isinstance(y_data, pd.DataFrame):
        raise TypeError("The KnnRegressor is only supported for multi-label regression.")

    # Drop nans in the single label approach
    x_data, y_data, X_metadata = _drop_missing_values_if_any(x_data, y_data, X_metadata)

    # Fit with or without hyperparameter tuning
    hpt_cv = hpt_tuning_cv_split is not None

    if hpt_cv:
        if hpt_tuning_param_grid is None:
            raise ValueError("A grid of hyper-parameters should be provided for the grid search.")
        if hpt_tuning_cv_split is None:
            raise ValueError("A split function for the hyper-parameter tuning should be provided.")

        # Define the grid of parameters to test
        param_grid_values = _define_param_grid(
            x_data=x_data,
            y_data=y_data,
            hpt_tuning_param_grid=hpt_tuning_param_grid,
            l1_ratio=(
                regression_model_base_instance.l1_ratio
                if isinstance(regression_model_base_instance, ElasticNet)
                else None
            ),
        )

        # Perform grid search
        grid = _param_search(
            param_grid=param_grid_values,
            regression_model_base_instance=regression_model_base_instance,
            x_data=x_data,
            y_data=y_data,
            X_metadata=X_metadata,
            hpt_tuning_cv_split=hpt_tuning_cv_split,
            hpt_tuning_score=hpt_tuning_score,
            param_search_type=param_search_type,
            ensembling=ensembling,
            ensembling_save_models_to_disk=ensembling_save_models_to_disk,
            ensembling_output_path=ensembling_output_path,
        )
    else:
        # Fit without CV - create a copy to avoid mutating the base instance
        model = copy.deepcopy(regression_model_base_instance)
        model.fit(x_data, y_data)
        grid = {
            "best_estimator_": model,
            "cv_results_": None,
        }

    return grid


def _fit_one_perturbation(
    x_data: pd.DataFrame,
    y_data: pd.Series | pd.DataFrame,
    X_metadata: pd.DataFrame,
    regression_model_base_instance: RegressionModel,
    hpt_tuning_param_grid: dict | None,
    hpt_tuning_cv_split: Callable | None,
    hpt_tuning_score: RegressionMetricType | None,
    param_search_type: dict | None = None,
    ensembling: bool = True,
    ensembling_save_models_to_disk: bool = False,
    ensembling_output_path: Path | None = None,
    pbar: tqdm | None = None,
) -> dict[str, Any]:
    """Fit a model for a single perturbation."""
    # Extract the label name
    label = str(y_data.name)
    if pbar is not None:
        pbar.set_description(f"Perturbation: {label:<25}")

    # Add the perturbation to the metadata
    X_metadata[PERTURBATION_INDEX] = label

    # Store the perturbation name in a multi index of all dataframes
    y_data = _add_perturbation_index(df=y_data, perturbation_name=label)
    x_data = _add_perturbation_index(df=x_data, perturbation_name=label)
    X_metadata = _add_perturbation_index(df=X_metadata, perturbation_name=label)

    # Used for stratification in binary classification tasks
    X_metadata["y"] = y_data

    # Do the hyper-parameter tuning with grid search
    return _fit_single_label(
        x_data=x_data,
        y_data=y_data,
        X_metadata=X_metadata,
        regression_model_base_instance=regression_model_base_instance,
        hpt_tuning_param_grid=hpt_tuning_param_grid,
        hpt_tuning_cv_split=hpt_tuning_cv_split,
        hpt_tuning_score=hpt_tuning_score,
        param_search_type=param_search_type,
        ensembling=ensembling,
        ensembling_save_models_to_disk=ensembling_save_models_to_disk,
        ensembling_output_path=ensembling_output_path,
    )


def define_model_params(
    param_grid: dict,
    param_search_type: dict | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Define model parameters.

    Parameters
    ----------
    param_grid : dict
        Dictionary with the parameters to search.
    param_search_type : dict | None
        Dictionary with the search type. Additional keys: if "search_type" is "random" are "n_models" which must be
        given and will lead to that number of sets of parameters being produced. "set_to_int" parameters which needs to
        be of type int instead of float, by default {"search_type": "grid"}.
    seed : int | None
        Seed for reproducibility, only used if "random" search type is being requested, by default None.

    Returns
    -------
    list[dict]
        List of parameter dictionaries.

    Raises
    ------
    ValueError
        If search_type is not 'grid' or 'random', or if 'n_models' is missing for random search.
    """
    if param_search_type is None:
        param_search_type = {"search_type": "grid"}
    if seed is not None:
        np.random.seed(seed)
    if param_search_type["search_type"] not in {"grid", "random"}:
        raise ValueError("search_type must be either 'grid' or 'random'.")

    parameter_names = list(param_grid.keys())

    if param_search_type["search_type"] == "grid":
        param_combinations: list[tuple] | np.ndarray = list(product(*param_grid.values()))
    else:
        if "n_models" not in param_search_type:
            raise ValueError("n_models must be defined for random search.")
        n_models = param_search_type["n_models"]

        # Extract parameter names, min, and max values
        min_values = np.array([np.min(param_grid[param]) for param in parameter_names])
        max_values = np.array([np.max(param_grid[param]) for param in parameter_names])

        # Generate n random models using uniform distribution between min and max
        param_combinations = np.random.uniform(min_values, max_values, size=(n_models, len(parameter_names)))

    # Convert each row of values into a dictionary
    models_params = [dict(zip(parameter_names, model_values, strict=False)) for model_values in param_combinations]

    # Convert selected parameters to int values
    if "set_to_int" in param_search_type:
        n_models = param_search_type.get("n_models", len(models_params))
        for key in set(param_search_type["set_to_int"]) & set(parameter_names):
            for i in range(n_models):
                models_params[i][key] = round(models_params[i][key])

    return models_params
