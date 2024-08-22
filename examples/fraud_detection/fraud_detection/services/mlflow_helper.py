"""This module contains MLflow related helper functions"""

from __future__ import annotations

from ae5_tools import load_ae5_user_secrets
from mlflow import MlflowClient

from fraud_detection.client.mlflow_client import AnacondaMlFlowClient


def mlflow_environment_config() -> AnacondaMlFlowClient:
    """
    Loads AE5 secrets, created and MLflow client, and ensures we have an experiment and model registry created.
    """

    # Load user specific configuration.
    load_ae5_user_secrets()

    mlflow_client: MlflowClient = MlflowClient()
    anaconda_mlflow_client = AnacondaMlFlowClient(client=mlflow_client)

    anaconda_mlflow_client.upsert_model_registry()

    return anaconda_mlflow_client
