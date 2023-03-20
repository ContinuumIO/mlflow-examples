import shlex
import subprocess
import uuid
import warnings
from pathlib import Path

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from .utils import build_run_name, process_launch_wait, upsert_experiment


@click.option("--worker-env-name", type=click.STRING, default="worker_env", help="The worker env name")
@click.option("--data-dir", type=click.STRING, default="data", help="The data directory")
@click.option(
    "--run-name", type=click.STRING, default="workflow-step-prepare-worker-environment", help="The name of the run"
)
@click.option("--unique", type=click.BOOL, default=True, help="Flag for appending a nonce to the end of run names")
@click.option("--backend", type=click.STRING, default="local", help="Flag for controlling logic for backend")
@click.command(help="Workflow Step [Pack Real-ESRGAN Runtime Environment]")
def run(worker_env_name: str, data_dir: str, run_name: str, unique: bool, backend: str):
    warnings.filterwarnings("ignore")
    with mlflow.start_run(nested=True, run_name=build_run_name(run_name=run_name, unique=unique)):
        if backend == "adsp" and not (Path(data_dir) / worker_env_name).exists():
            # Pack Worker Runtime Environment
            cmd: str = "anaconda-project run bootstrap"
            process_launch_wait(shell_out_cmd=cmd)
        else:
            print("Skipping worker environment preparation, either wrong backend or already complete")


if __name__ == "__main__":
    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
