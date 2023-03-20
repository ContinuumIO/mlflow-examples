""" This module contains common utilities used by the workflow steps. """

import os
import secrets
import shlex
import string
import subprocess
from typing import List, Optional

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


def build_run_name(run_name: str, unique: bool) -> str:
    """
    Given a run name and a uniqueness flag will generate a run name.

    Parameters
    ----------
    run_name: str
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
        return run_name + "-" + unique_suffix
    return run_name


def process_launch_wait(shell_out_cmd: str, cwd: str = ".") -> None:
    """
    Internal function for wrapping process launches [and waiting].

    Parameters
    ----------
    shell_out_cmd: str
        The command to be executed.
    cwd: str
        The `current working directory` of the command.  This is the directory to launch the command from.
    """

    args = shlex.split(shell_out_cmd)

    with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE) as process:
        for line in iter(process.stdout.readline, b""):
            print(line)


def get_batches(batch_size: int, source_list: List[str]) -> List[List[str]]:
    """
    Given a batch size and a list of strings, will generate a list of lists of strings split by batch size.

    Parameters
    ----------
    batch_size: int
        The maximum size a batch can be.  It is possible to receive a batch that is smaller than this.
        This value must be greater than zero.
    source_list: List[str]
        A list of strings to split into batches.

    Returns
    -------
    batches: List[List[str]]
        The source list split up by batch size into smaller lists and returned together.
    """

    if batch_size < 1:
        raise ValueError(f"Batch size must be greater than zero.  Saw: ({batch_size})")

    batches: List = []

    while len(source_list) > 0:
        if len(source_list) >= batch_size:
            new_batch: List[str] = source_list[:batch_size]
            source_list[:batch_size] = []
        else:
            new_batch: List[str] = source_list
            source_list = []
        batches.append(new_batch)

    return batches
