"""
Workflow Step [Process One] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m workflow.steps.process_one`
When invoked this way the click defaults are used.

2. MLFlow CLI:
`mlflow run . -e process_one --env-manager local`
When invoked this way the MLproject default parameters are used.
"""

import warnings

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from mlflow_adsp import create_unique_name, upsert_experiment


@click.command(help="Process One")
@click.option("--some-parameter-int", type=click.INT, default=1, help="An Integer Parameter")
@click.option("--some-parameter-float", type=click.FLOAT, default=1.0, help="A Float Parameter")
@click.option("--some-parameter-string", type=click.STRING, default="1", help="A String Parameter")
@click.option(
    "--run-name", type=click.STRING, default="template-project-workflow-step-process-one", help="The name of the run"
)
def run(some_parameter_int: int, some_parameter_float: float, some_parameter_string: str, run_name: str) -> None:
    """
    Workflow Step [Process One] Entry Point

    Parameters
    ----------
    some_parameter_int: int
        An Integer Parameter
    some_parameter_float: float
        A Float Parameter
    some_parameter_string: str
        A String Parameter
    run_name: str
        Default: `template-project-workflow-step-process-one`
        The name of the run.
    """

    warnings.filterwarnings("ignore")
    with mlflow.start_run(nested=True, run_name=create_unique_name(name=run_name)):
        mlflow.log_param(key="some_parameter_int", value=some_parameter_int)
        mlflow.log_param(key="some_parameter_float", value=some_parameter_float)
        mlflow.log_param(key="some_parameter_string", value=some_parameter_string)

        mlflow.log_dict(dictionary={"sample_key": "sample_value"}, artifact_file="business_metrics.json")


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
