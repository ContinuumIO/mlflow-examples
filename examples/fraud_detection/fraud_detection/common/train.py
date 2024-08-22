"""
ML Utility functions:

- `gridsearch_train`:
    * Runs a GridSearchCV training pipeline for a given ML model (Classifier) & its params on Training data partition, i.e. `(X_train, y_train)`.
    By default, the hyper-parameter search is applied within a `10x5RepeatedStratifiedCV` framework.
    The `F1` measure is used as it's the most appropriate metric given the problem at hand (Anomaly detection).
    Preprocessing pipeline steps are optional, along with its corresponding hyper-parameters.
- `train`:
    * Calls `gridsearch_train` and `evaluate` functions in turn for a given model/preprocessing pipeline.
"""

import logging
from datetime import datetime, timedelta

import mlflow

# Pipeline
# (imblearn Pipeline because we need to embed a sampler)
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Model Selection and Metrics
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from fraud_detection.common.constants import rng
from fraud_detection.contracts.dto.ml_model_config import MLModelConfig
from fraud_detection.contracts.dto.training_params import TrainingParams
from fraud_detection.report.visual import visualize_tree

logger = logging.getLogger(__name__)


def gridsearch_train(train_params: TrainingParams) -> GridSearchCV:
    """
    Train a given model (with Hyper Parameter Tuning) within a Repeated Stratified 10x5CV

    Parameters
    ----------
    train_params: TrainingParams
        Model Config

    Returns
    -------
    gs_model: GridSearchCV
        The training instance of the GridSearchCV object.
    """
    train_params.ml_model_config = MLModelConfig.model_validate(train_params.ml_model_config)

    pipeline: Pipeline = Pipeline(train_params.preprocessing_steps + [("model", train_params.ml_model_config.model)])

    if train_params.preprocessing_params is not None:
        pipeline_params = train_params.preprocessing_params | train_params.ml_model_config.params
    else:
        pipeline_params = train_params.ml_model_config.params
    logger.info("Training: %s", train_params.ml_model_config.name)
    logger.info("Params: %s", pipeline_params)
    logger.info("Pipeline: %s", pipeline)

    gs_model: GridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_params,
        n_jobs=-1,
        scoring="f1",
        cv=RepeatedStratifiedKFold(n_repeats=train_params.cv_n_reps, n_splits=train_params.cv_n_splits, random_state=rng),
    )
    gs_model.fit(train_params.dataset.X_train, train_params.dataset.y_train)

    # Generate additional reports

    # This builds the visualizations for decision trees and random forests
    dt_model = gs_model.best_estimator_.named_steps["model"]
    feature_names = gs_model.feature_names_in_
    if isinstance(train_params.ml_model_config.model, RandomForestClassifier):
        for index, tree in enumerate(dt_model.estimators_):
            visualize_tree(tree=tree, name=f"random_forest_{index}", feature_names=feature_names)
    elif isinstance(train_params.ml_model_config.model, DecisionTreeClassifier):
        visualize_tree(tree=dt_model, name="decision_tree_1", feature_names=feature_names)

    logger.info("Best Params: %s", gs_model.best_params_)
    logger.info("Best CV Score (F1): %s", gs_model.best_score_)

    return gs_model


def train(train_params: TrainingParams) -> dict:
    """
    Wraps call to model-specific training function `gridsearch_train` and returns tags for model registration.

    Parameters
    ----------
    train_params: TrainingParams
        Model Config

    Returns
    -------
        tags: dict
            Models tags suitable for use in model registration.
    """

    # train
    start: datetime = datetime.now()
    gs_model: GridSearchCV = gridsearch_train(train_params=train_params)
    elapsed: timedelta = datetime.now() - start

    mlflow.log_metric(key="total_train_time", value=elapsed.total_seconds())
    logger.info("Elapsed Time to run %ix%iCV: %f seconds", train_params.cv_n_reps, train_params.cv_n_splits, elapsed.total_seconds())

    # Tags related to the completed process (used by registration after this call)
    return {
        **train_params.ml_model_config.params,
        "best_cv_score_f1": gs_model.best_score_,
        "model_type": train_params.ml_model_config.name.lower().strip().replace(" ", "_"),
    }
