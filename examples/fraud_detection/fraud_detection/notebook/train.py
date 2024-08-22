# pylint: skip-file
"""
ML Utility functions:

- `train`:
    * Runs a GridSearchCV training pipeline for a given ML model (Classifier) & its params on Training data partition, i.e. `(X_train, y_train)`.
    By default, the hyper-parameter search is applied within a `10x5RepeatedStratifiedCV` framework.
    The `F1` measure is used as it's the most appropriate metric given the problem at hand (Anomaly detection).
    Preprocessing pipeline steps are optional, along with its corresponding hyper-parameters.
- `evaluate`:
    * Evaluate a model on external (held out) Test data partition, i.e. `(X_test, y_test)`, using the `F1` metric as
    evaluation metric/criterion.
- `run_experiment`:
    * Calls `train` and `evaluate` functions in turn for a given model/preprocessing pipeline.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import mlflow
import numpy as np

# Pipeline
# (imblearn Pipeline because we need to embed a sampler)
from imblearn.pipeline import Pipeline

# Model Persistence
from joblib import dump
from mlflow_adsp import create_unique_name
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score

# Model Selection and Metrics
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from fraud_detection.services.mlflow_helper import mlflow_environment_config

# MLflow - Enable Autologging for sklearn
mlflow.sklearn.autolog()

# Init our MLflow experiment
mlflow_environment_config()


# Inspired from https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
class Estimator(Protocol):
    def fit(self, X, y, sample_weight=None): ...

    def predict(self, X): ...

    def score(self, X, y, sample_weight=None): ...

    def set_params(self, **params): ...


ModelInfo = tuple[str, Estimator, dict[str, list[Any]]]
PipelineSteps = list[tuple[str, Estimator]]
Params = dict[str, Any]
Partition = tuple[ArrayLike, ArrayLike]


MODEL_FOLDER = Path(os.environ["MODELS_BASE_DIR"])
os.makedirs(MODEL_FOLDER, exist_ok=True)


def train(
    model_info: ModelInfo,
    preprocessing_steps: PipelineSteps,
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    preprocessing_params: Params = None,
    rng: np.random.RandomState = None,
    cv_n_reps: int = 10,
    cv_n_splits: int = 5,
    verbose: bool = True,
) -> Estimator:
    """Train a given model (with Hyper Parameter Tuning) within a Repeated Stratified 10x5CV"""

    model_name, model, model_params = model_info
    pipeline = Pipeline(preprocessing_steps + [("model", model)])

    if preprocessing_params is not None:
        pipeline_params = preprocessing_params | model_params
    else:
        pipeline_params = model_params

    print(f"Training {model_name}")
    if verbose:
        print(f"Params: {pipeline_params}")
        print(f"Pipeline: {pipeline}")

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_params,
        n_jobs=-1,
        scoring="f1",
        cv=RepeatedStratifiedKFold(n_repeats=cv_n_reps, n_splits=cv_n_splits, random_state=rng),
    )

    with mlflow.start_run(nested=True, run_name=create_unique_name(name="fraud_detection_demo")):
        gs.fit(X_train, y_train)

    if verbose:
        print("Best Params: ", gs.best_params_)
        print("Best CV Score (F1)", gs.best_score_)

    return gs


def evaluate(model_name: str, model: Estimator, X_test: ArrayLike, y_test: ArrayLike) -> float:
    print(f"Evaluate {model_name}")
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)


def run_experiment(
    name: str,
    model_configs: list[ModelInfo],
    data: Partition,
    labels: Partition,
    preprocessing_steps: PipelineSteps,
    *,
    cv_n_reps: int = 10,
    cv_n_splits: int = 5,
    preproc_hyper_params: Params = None,
    rng: np.random.RandomState = None,
):
    """
    Run the full experiment on selected models (and Params),
    calling train and evaluate, in turn.
    """
    X_train, X_test = data
    y_train, y_test = labels
    exp_label = name.lower().strip().replace(" ", "_")

    for model_info in model_configs:
        start = datetime.now()
        gs_model = train(
            model_info=model_info,
            preprocessing_steps=preprocessing_steps,
            preprocessing_params=preproc_hyper_params,
            X_train=X_train,
            y_train=y_train,
            cv_n_reps=cv_n_reps,
            cv_n_splits=cv_n_splits,
            rng=rng,
            verbose=True,
        )
        elapsed = datetime.now() - start
        print(f"Elapsed Time to run {cv_n_reps}x{cv_n_splits}CV: {elapsed} seconds")
        best_model = gs_model.best_estimator_
        model_name, *_ = model_info
        print(evaluate(model_name=model_name, model=best_model, X_test=X_test, y_test=y_test))
        model_filename = f"gs_{model_name.lower().replace(' ', '_')}_{exp_label}.joblib"
        model_filepath = MODEL_FOLDER / model_filename
        dump(gs_model, model_filepath)
        print("")  # Empty line, mostly for clean report
