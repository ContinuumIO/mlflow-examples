"""
Workflow Step [Main] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m workflow.steps.main`
When invoked this way the click defaults are used.

2. MLFlow CLI:
`mlflow run . --env-manager local`
When invoked this way the MLproject default parameters are used.
"""

import math
import uuid
from pathlib import Path
from typing import List

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from mlflow_adsp import create_unique_name, upsert_experiment


@click.command(help="Workflow [Main]")
@click.option("--some-parameter-int", type=click.INT, default=1, help="An Integer Parameter")
@click.option("--some-parameter-float", type=click.FLOAT, default=1.0, help="A Float Parameter")
@click.option("--some-parameter-string", type=click.STRING, default="1", help="A String Parameter")
@click.option(
    "--run-name", type=click.STRING, default="template-project-workflow-main", help="The name of the run"
)
def workflow(some_parameter_int: int, some_parameter_float: float, some_parameter_string: str, run_name: str) -> None:
    """
    Workflow Entry Point

    Parameters
    ----------
    some_parameter_int: int
        An Integer Parameter
    some_parameter_float: float
        A Float Parameter
    some_parameter_string: str
        A String Parameter
    run_name: str
        Default: `template-project-workflow-main`
        The name of the run.
    """

    with mlflow.start_run(run_name=create_unique_name(name=run_name)) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #

        #############################################################################
        # Set up runtime environment
        #############################################################################

        print(f"some_parameter_int={some_parameter_int}")
        print(f"some_parameter_float={some_parameter_float}")
        print(f"some_parameter_string={some_parameter_string}")

        run_id: str = run.info.run_id
        print(f"run_id: {run_id}")

        #############################################################################
        # Execute workflow steps
        #############################################################################

        mlflow.projects.run(**{
            "entry_point": "process_one",
            "parameters": {
                "some_parameter_int": some_parameter_int,
                "some_parameter_float": some_parameter_float,
                "some_parameter_string": some_parameter_string
            },
            "run_name": create_unique_name(name="template-project-workflow-step-process-one"),
            "uri": ".",
            "env_manager": "local"
        })


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    workflow()
