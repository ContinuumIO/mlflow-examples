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


def get_best_run(client: MlflowClient, experiment_id, runs: list[str]) -> tuple[Optional[Run], dict]:
    _inf = np.finfo(np.float64).max

    best_metrics: dict = {
        "validation_0-rmse": _inf,
    }
    best_run: Optional[Run] = None

    for run_id in runs:
        # find the best run, log its metrics as the final metrics of this run.
        run: Run = client.search_runs([experiment_id], f"attributes.run_id = '{run_id}'")[0]
        if (
            "validation_0-rmse" in run.data.metrics
            and run.data.metrics["validation_0-rmse"] < best_metrics["validation_0-rmse"]
        ):
            best_metrics = run.data.metrics
            best_run = run

    return best_run, best_metrics
