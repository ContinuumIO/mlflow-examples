import os
from typing import Optional

import numpy as np
from mlflow import MlflowClient, MlflowException
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


def upsert_model_registry(client: MlflowClient) -> None:
    try:
        client.create_registered_model(name=os.environ["MLFLOW_EXPERIMENT_NAME"])
    except (MlflowException, RestException) as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise error


def register_best_model(client: MlflowClient, run: Run) -> ModelVersion:
    model_version: ModelVersion = client.create_model_version(
        name=os.environ["MLFLOW_EXPERIMENT_NAME"],
        source=f"{run.info.artifact_uri}/model",
        run_id=run.info.run_id,
        tags={"run_id": run.info.run_id},
    )
    return model_version
