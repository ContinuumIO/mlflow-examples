""" MLFlow Tracking Server Helpers """

import os
import secrets
import string
from typing import Optional

import mlflow
from mlflow.entities import Experiment


def upsert_experiment(experiment_name: str = "Default") -> str:
    """
    This function returns the experiment id for the provided experiment name.
    If the experiment does not exist, it is created.

    This function looks for an environment variable called `AE_MLFLOW_EXPERIMENT_NAME`.
    If one is defined the value is used REGARDLESS of whether the caller provided
    an experiment name.

    Parameters
    ----------
    experiment_name: str
        The experiment name.

    Returns
    -------
    experiment_id: str
        The experiment Id.
    """

    # Env var will over-ride
    if "AE_MLFLOW_EXPERIMENT_NAME" in os.environ:
        experiment_name = os.environ["AE_MLFLOW_EXPERIMENT_NAME"]

    try:
        experiment: Optional[Experiment] = mlflow.get_experiment_by_name(name=experiment_name)
        experiment_id: str = experiment.experiment_id
    # I do not know what types of exceptions to expect from mlflow.get_experiment_by_name, it is not documented.
    # pylint: disable=broad-exception-caught
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


def build_run_name(name: str, unique: bool) -> str:
    """
    Given a run name and a uniqueness flag will generate a run name.

    Parameters
    ----------
    name: str
        The original run name.
    unique: bool
        Flag for making the name unique.

    Returns
    -------
    run_name: str
        Returns the run name (possibly mutated based on uniqueness flag).
    """

    if unique:
        alphabet = string.ascii_letters + string.digits
        unique_suffix: str = "".join(secrets.choice(alphabet) for i in range(10))
        return name + "-" + unique_suffix
    return name
