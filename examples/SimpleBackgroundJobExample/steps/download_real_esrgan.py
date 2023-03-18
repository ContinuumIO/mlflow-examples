import shlex
import subprocess
import uuid
import warnings
from pathlib import Path

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from .utils import build_run_name, process_launch_wait, upsert_experiment


# Note: If run stand alone (just the step) the run will report to a new job,
# rather than under a parent job (since one does not exist).
@click.option("--source", default="https://github.com/xinntao/Real-ESRGAN.git", type=click.STRING)
@click.option(
    "--source-dir", type=click.STRING, default="data/Real-ESRGAN", help="The source directory for real-esrgran"
)
@click.option("--run-name", type=click.STRING, default="workflow-step-download-real-esrgan", help="The name of the run")
@click.option("--unique", type=click.BOOL, default=True, help="Flag for appending a nonce to the end of run names")
@click.command(help="Workflow Step [Download Real-ESRGAN]")
def run(source: str, source_dir: str, run_name: str, unique: bool):
    warnings.filterwarnings("ignore")

    source_path: Path = Path(source_dir)

    with mlflow.start_run(nested=True, run_name=build_run_name(run_name=run_name, unique=unique)):
        if source_path.exists() and source_path.is_dir():
            cmd: str = f"cd {source_path.as_posix()} && git pull"
        else:
            cmd: str = f"git clone --depth 1 --single-branch --no-tags {source} {source_path}"
        process_launch_wait(shell_out_cmd=cmd, cwd=".")

        # Setup dependencies
        cmd: str = "python setup.py develop"
        process_launch_wait(shell_out_cmd=cmd, cwd=source_path.resolve())


if __name__ == "__main__":
    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
