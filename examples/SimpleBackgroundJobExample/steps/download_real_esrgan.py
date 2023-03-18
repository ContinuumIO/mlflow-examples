import shlex
import subprocess
import uuid
import warnings
from pathlib import Path

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from .utils import build_run_name, upsert_experiment


def process_launch_wait(shell_out_cmd: str) -> None:
    """
    Internal function for wrapping process launches [and waiting].

    Parameters
    ----------
    shell_out_cmd: str
        The command to be executed.
    """

    args = shlex.split(shell_out_cmd)

    with subprocess.Popen(args, stdout=subprocess.PIPE) as process:
        for line in iter(process.stdout.readline, b""):
            print(line)


# Note: If run stand alone (just the step) the run will report to a new job,
# rather than under a parent job (since one does not exist).
@click.option("--source", default="https://github.com/xinntao/Real-ESRGAN.git", type=click.STRING)
@click.option("--work-dir", type=click.STRING, default="data", help="The base directory to work within")
@click.option("--run-name", type=click.STRING, default="workflow-step-download-real-esrgan", help="The name of the run")
@click.option("--unique", type=click.BOOL, default=True, help="Flag for appending a nonce to the end of run names")
@click.command(help="Workflow Step [Download Real-ESRGAN]")
def run(source: str, work_dir: str, run_name: str, unique: bool):
    warnings.filterwarnings("ignore")

    source_dir: Path = Path(work_dir) / "Real-ESRGAN"

    with mlflow.start_run(nested=True, run_name=build_run_name(run_name=run_name, unique=unique)):
        if source_dir.exists() and source_dir.is_dir():
            cmd: str = f"cd {source_dir.as_posix()} && git pull"
        else:
            cmd: str = f"git clone --depth 1 --single-branch --no-tags {source} {source_dir}"
        process_launch_wait(shell_out_cmd=cmd)


if __name__ == "__main__":
    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
