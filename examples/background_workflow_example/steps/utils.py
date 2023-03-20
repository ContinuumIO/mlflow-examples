import os
import secrets
import shlex
import string
import subprocess
from typing import List, Optional

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


def process_launch_wait(shell_out_cmd: str, cwd: str = ".") -> None:
    """
    Internal function for wrapping process launches [and waiting].

    Parameters
    ----------
    shell_out_cmd: str
        The command to be executed.
    """

    args = shlex.split(shell_out_cmd)

    with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE) as process:
        for line in iter(process.stdout.readline, b""):
            print(line)


def get_batches(batch_size: int, file_list: List[str]) -> List:
    if batch_size < 1:
        raise ValueError(f"Batch size must be greater than zero.  Saw: ({batch_size})")

    batches: List = []

    while len(file_list) > 0:
        if len(file_list) >= batch_size:
            new_batch: List[str] = file_list[:batch_size]
            file_list[:batch_size] = []
        else:
            new_batch: List[str] = file_list
            file_list = []
        batches.append(new_batch)

    return batches
