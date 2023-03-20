"""
Workflow Step [Download Real ESRGAN] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m steps.download_real_esrgan`
When invoked this way the click defaults are used.

2. MLFlow CLI:
`mlflow run . -e download_real_esrgran`
When invoked this way the MLproject default parameters are used

3. Workflow (or other code)
The function and its set up can be called from other code.
The `main` step does this in the workflow definition.

Note:
    If run stand alone (just the step) the run will report to a new job,
    rather than under a parent job (since one does not exist).

"""

import warnings
from pathlib import Path

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from .utils import build_run_name, process_launch_wait, upsert_experiment


@click.option("--source", default="https://github.com/xinntao/Real-ESRGAN.git", type=click.STRING)
@click.option(
    "--source-dir", type=click.STRING, default="data/Real-ESRGAN", help="The source directory for real-esrgran"
)
@click.option("--run-name", type=click.STRING, default="workflow-step-download-real-esrgan", help="The name of the run")
@click.option("--unique", type=click.BOOL, default=True, help="Flag for appending a nonce to the end of run names")
@click.command(help="Workflow Step [Download Real-ESRGAN]")
def run(source: str, source_dir: str, run_name: str, unique: bool) -> None:
    """

    Parameters
    ----------
    source: str
        The git source repo for the real-esrgan code base.
    source_dir: str
        The directory to checkout the git repo into.
    run_name: str
        The base name of the run (for reporting to MLFlow)
    unique: bool
        Flag to control whether to make the provided name unique.
    """

    warnings.filterwarnings("ignore")

    source_path: Path = Path(source_dir)

    with mlflow.start_run(nested=True, run_name=build_run_name(run_name=run_name, unique=unique)):
        # Download or update our framework
        if source_path.exists() and source_path.is_dir():
            cmd: str = f"cd {source_path.as_posix()} && git pull"
        else:
            cmd: str = f"git clone --depth 1 --single-branch --no-tags {source} {source_path}"
        process_launch_wait(shell_out_cmd=cmd, cwd=".")

        # Setup dependencies for framework
        cmd: str = "python setup.py develop"
        process_launch_wait(shell_out_cmd=cmd, cwd=source_path.resolve())


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
