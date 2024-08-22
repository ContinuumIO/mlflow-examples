"""
Workflow Step [Process Data] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m workflow.steps.process_data`
When invoked this way the click defaults are used.

2. Workflow (or other code)
The function and its set up can be called from other code.
The `main` step does this in the workflow definition.

Note:
    If run stand alone (just the step) the run will report to a new job,
    rather than under a parent job (since one does not exist).
"""

import os
from pathlib import Path

import click
import mlflow
from ae5_tools import demand_env_var
from mlflow_adsp import create_unique_name

from fraud_detection.client.data import DataClient
from fraud_detection.client.mlflow_client import AnacondaMlFlowClient
from fraud_detection.common.train import train
from fraud_detection.contracts.dto.dataset import DataSet
from fraud_detection.contracts.dto.training_params import TrainingParams
from fraud_detection.contracts.types.model import ModelType
from fraud_detection.contracts.types.sampling_strategy import SamplingStrategyType
from fraud_detection.services.mlflow_helper import mlflow_environment_config
from fraud_detection.services.model_factory import ModelFactory


@click.command(help="Workflow Step [Train]")
@click.option(
    "--run-name",
    type=click.STRING,
    default="workflow-step-train",
    help="The base name of the run (for reporting to MLFlow).",
)
@click.option(
    "--model-type",
    type=click.STRING,
    default="decision_tree",
)
@click.option(
    "--strategy",
    type=click.STRING,
    default="under_sampling_near_miss",
)
@click.option(
    "--cv-n-reps",
    type=click.INT,
    default=10,
)
@click.option(
    "--cv-n-splits",
    type=click.INT,
    default=5,
)
def workflow_step_train(run_name: str, model_type: str, strategy: str, cv_n_reps: int = 10, cv_n_splits: int = 5) -> None:
    """
    Runs the Workflow Step ['Worker' Training]

    Parameters
    ----------
    run_name: str
        The base name of the run (for reporting to MLFlow).
    model_type: str
        The model type to train. (Supported by the factory)
    strategy: list[str]
        The sampling strategy to use during training. (Supported by the factory)
    cv_n_reps: int
        Number of times cross-validator needs to be repeated.
    cv_n_splits: int
        Number of folds. Must be at least 2.
    """

    # MLflow - Enable Autologging for sklearn
    mlflow.sklearn.autolog()

    # Init our MLflow experiment environment
    mlflow_helper: AnacondaMlFlowClient = mlflow_environment_config()

    data_client: DataClient = DataClient(
        csv_url=Path(os.getenv("DATA_BASE_DIR")) / os.getenv("DATA_ARTIFACT"), truth_col_name=demand_env_var(name="TRUTH_COLUMN_NAME")
    )
    dataset: DataSet = data_client.get_dataset()

    with mlflow.start_run(nested=True, run_name=create_unique_name(name=run_name)) as run:
        model_factory: ModelFactory = ModelFactory(model_type=ModelType(model_type), strategy=SamplingStrategyType(strategy))

        training_params: TrainingParams = TrainingParams(
            ml_model_config=model_factory.build_model(),
            **model_factory.build_model_preprocessing_steps(),
            dataset=dataset,
            cv_n_reps=cv_n_reps,
            cv_n_splits=cv_n_splits,
        )
        tags: dict = train(training_params)
        mlflow_helper.register_model(run=run, tags=tags, suffix="best_estimator")


if __name__ == "__main__":
    workflow_step_train()
