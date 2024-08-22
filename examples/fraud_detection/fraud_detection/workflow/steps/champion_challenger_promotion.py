"""
Workflow Step [Champion Challenger Promotion] Definition

This step can be invoked in two different ways:
1. Python module invocation:
`python -m fraud_detection.workflow.steps.champion_challenger_promotion --run-name {run_name}`
When invoked this way the click defaults are used.

2. Workflow (or other code)
The function and its set up can be called from other code.
The `main` step does this in the workflow definition.

Note:
    If run stand alone (just the step) the run will report to a new job,
    rather than under a parent job (since one does not exist).
"""

import logging
import os
from pathlib import Path

import click
import mlflow
from ae5_tools import demand_env_var, demand_env_var_as_bool
from mlflow import MlflowClient
from mlflow_adsp import create_unique_name

from fraud_detection.client.data import DataClient
from fraud_detection.report.mlflow_reporting import MLflowReporting
from fraud_detection.services.mlflow_helper import mlflow_environment_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(help="Workflow Step [Champion Challenger Promotion]")
@click.option(
    "--run-name",
    type=click.STRING,
    default="champion-challenger-promotion",
    help="The base name of the run (for reporting to MLFlow).",
)
@click.option(
    "--parent-run-id",
    type=click.STRING,
    help="The MLflow parent workflow run id for child run analysis.",
)
def champion_challenger_promotion(run_name: str, parent_run_id: str) -> None:
    """
    Runs the Workflow Step ['Worker' Training]

    Parameters
    ----------
    run_name: str
        The base name of the run (for reporting to MLFlow).
    parent_run_id: str
        The MLflow parent workflow run id for child run analysis.
    """

    # Init our MLflow experiment environment

    mlflow_client: MlflowClient = mlflow_environment_config().client

    # Create our report builder
    mlflow_reporting: MLflowReporting = MLflowReporting(client=mlflow_client)

    # Data client for champion/challenger report
    data_client: DataClient = DataClient(
        csv_url=Path(os.getenv("DATA_BASE_DIR")) / os.getenv("DATA_ARTIFACT"), truth_col_name=demand_env_var(name="TRUTH_COLUMN_NAME")
    )

    with mlflow.start_run(nested=True, run_name=create_unique_name(name=run_name)):
        if demand_env_var_as_bool(name="SELF_HOSTED_MODEL_AUTO_PROMOTION"):
            version: str = mlflow_reporting.champion_challenger_report(run_id=parent_run_id, data_client=data_client)
            # Then the local winner is the global winner.  Promote..
            mlflow_client.set_registered_model_alias(
                name=demand_env_var(name="MLFLOW_EXPERIMENT_NAME"), alias=demand_env_var(name="SELF_HOSTED_MODEL_ALIAS"), version=version
            )
            logger.info("Version %s is Champion", version)
        else:
            logger.info("Auto promotion is disabled, skipping step..")


if __name__ == "__main__":
    champion_challenger_promotion()
