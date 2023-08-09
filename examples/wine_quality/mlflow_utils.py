"""
This module contains MLflow related helper functions.
"""

import os

from mlflow import MlflowClient, MlflowException
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


def upsert_model_registry(client: MlflowClient) -> None:
    """
    Upsert (Update or Insert) a model registry into MLflow.

    Parameters
    ----------
    client: MlflowClient
        Instance of an MLflow client.
    """

    try:
        client.create_registered_model(name=os.environ["MLFLOW_EXPERIMENT_NAME"])
    except (MlflowException, RestException) as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise error


def register_best_model(client: MlflowClient, run: Run) -> ModelVersion:
    """
    Registers the model tracked on the provided run.

    Parameters
    ----------
    client: MlflowClient
        Instance of an MLflow client.
    run: Run
        Instance of an MLflow Run with a model logged to it.

    Returns
    -------
    version: ModelVersion
        The model version created during registration.
    """

    model_version: ModelVersion = client.create_model_version(
        name=os.environ["MLFLOW_EXPERIMENT_NAME"],
        source=f"{run.info.artifact_uri}/model",
        run_id=run.info.run_id,
        tags={"run_id": run.info.run_id},
    )
    return model_version
