""" This module contains common utilities used by the workflow steps. """

import os
import secrets
import shlex
import string
import subprocess
import time
from typing import Dict, List, Optional

import mlflow
from mlflow.entities import Experiment, RunStatus
from mlflow_adsp import AnacondaEnterpriseSubmittedRun


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


def get_status_of_workers(workers: List[AnacondaEnterpriseSubmittedRun]) -> List[RunStatus]:
    statuses: List[RunStatus] = []

    for worker in workers:
        statuses.append(worker.get_status())

    return statuses


def check_workers_complete(workers: List[AnacondaEnterpriseSubmittedRun]) -> bool:
    statuses: List[RunStatus] = get_status_of_workers(workers=workers)

    running_workers: int = 0
    for status in statuses:
        if status == RunStatus.RUNNING:
            running_workers += 1

    if running_workers != 0:
        print(f"{running_workers} of {len(workers)} workers still running")
        return False

    return True


def wait_on_workers(workers: List[AnacondaEnterpriseSubmittedRun]) -> None:
    wait_time: int = 10  # seconds
    max_wait_counter: int = 100
    counter: int = 0
    wait: bool = True

    while (wait):
        if check_workers_complete(workers=workers):
            wait = False
        else:
            time.sleep(wait_time)
            counter += 1
        if counter >= max_wait_counter:
            wait = False

    if counter >= max_wait_counter:
        raise Exception("Not all workers completed within the defined time frame")



# def batch_process(parameters: Dict = {}) -> List:
#     workers: List = []
#     worker_count: int = 2
#     unique: bool = True
#
#     with mlflow.start_run(run_name=build_run_name(run_name="manual-workflow-invocation", unique=unique)) as run:
#         for i in range(0, worker_count):
#             print(i)
#             background_job = mlflow.projects.run(
#                 uri=".",
#                 entry_point="process_one",
#                 run_name=build_run_name(run_name="manual-workflow-worker", unique=unique),
#                 env_manager="local",
#                 backend="adsp",
#                 parameters=parameters,
#                 synchronous=False
#             )
#             workers.append(background_job)
#
#     return workers