"""
This module contains environmental helper functions.
"""

import mlflow
import mlflow.sklearn
from mlflow.entities import Experiment
from mlflow.tracking import MlflowClient

from ae5_tools import load_ae5_user_secrets
from mlflow_adsp import upsert_experiment
from mlflow_utils import upsert_model_registry


def init() -> tuple[str, MlflowClient]:
    """
    Loads AE5 secrets, created and MLflow client, and ensures we have an experiment and model registry created.

    Returns
    -------
    tuple
        A tuple of (experiment id, MLflow client).
    """

    # Load user specific configuration.
    load_ae5_user_secrets()

    # Generate a client, this will be used for several operations across the notebook.
    client = MlflowClient()

    experiment: Experiment = mlflow.set_experiment(experiment_id=upsert_experiment())
    upsert_model_registry(client=client)

    return experiment.experiment_id, client
