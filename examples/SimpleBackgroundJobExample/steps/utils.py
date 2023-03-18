import os
import secrets
import string
from typing import Optional

import mlflow
from mlflow.entities import Experiment


def upsert_experiment(experiment_name: str = "Default") -> str:
    # Env var will over-ride
    if "AE_MLFLOW_EXPERIMENT_NAME" in os.environ:
        experiment_name = os.environ["AE_MLFLOW_EXPERIMENT_NAME"]

    try:
        experiment: Optional[Experiment] = mlflow.get_experiment_by_name(name=experiment_name)
        experiment_id: str = experiment.experiment_id
    except Exception as error:
        print(error)
        print("Creating experiment")

        try:
            experiment_id: str = mlflow.create_experiment(name=experiment_name)
        except Exception as inner_error:
            print(f"Unable to create experiment with the name {experiment_name}")
            print(error)
            raise inner_error

    return experiment_id


def build_run_name(run_name: str, unique: bool) -> str:
    if unique:
        alphabet = string.ascii_letters + string.digits
        unique_suffix: str = "".join(secrets.choice(alphabet) for i in range(10))
        return run_name + "-" + unique_suffix
    return run_name
