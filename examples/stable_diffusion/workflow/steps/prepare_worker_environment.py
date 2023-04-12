"""
Workflow Step [Worker Runtime Setup] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m workflow.steps.prepare_worker_environment`
When invoked this way the click defaults are used.

2. MLFlow CLI:
`mlflow run .  -e prepare_worker_environment` --backend local`
- or -
`mlflow run .  -e prepare_worker_environment` --backend adsp`
When invoked this way the MLproject default parameters are used.

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
from mlflow_adsp import create_unique_name, upsert_experiment

from ..utils.process import process_launch_wait


@click.option("--worker-env-name", type=click.STRING, default="worker_env", help="The worker environment name.")
@click.option("--data-dir", type=click.STRING, default="data", help="The shared storage directory base.")
@click.option(
    "--run-name",
    type=click.STRING,
    default="workflow-step-prepare-worker-environment",
    help="The name of the run to use for reporting.",
)
@click.option(
    "--backend",
    type=click.STRING,
    default="local",
    help=(
        "The backend type for run context. "
        "We only pack when we are targeting the `adsp` backend, all others are skipped."
    ),
)
@click.command(help="Workflow Step [Prepare Runtime Environment]")
def run(worker_env_name: str, data_dir: str, run_name: str, backend: str) -> None:
    """
    Runs the worker bootstrap within a mlflow job.
    If the worker environment has previously been created within the shared location it will NOT be recreated.

    Parameters
    ----------
    worker_env_name: str
        The worker environment name.
    data_dir: str
        The shared storage directory base.
    run_name: str
        The name of the run to use for reporting.
    backend: str
        The backend type for run context.
        We only pack when we are targeting the `adsp` backend, all others are skipped.
    """

    warnings.filterwarnings("ignore")
    with mlflow.start_run(nested=True, run_name=create_unique_name(name=run_name)):
        if backend == "adsp" and not (Path(data_dir) / worker_env_name).exists():
            # Pack Worker Runtime Environment
            cmd: str = "anaconda-project run bootstrap"
            process_launch_wait(shell_out_cmd=cmd)
        else:
            print("Skipping worker environment preparation, either wrong backend or already complete")


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
