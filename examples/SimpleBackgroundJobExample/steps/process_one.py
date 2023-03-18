import json
import warnings
from typing import Dict

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from .utils import build_run_name, upsert_experiment


# Note: If run stand alone (just the step) the run will report to a new job,
# rather than under a parent job (since one does not exist).
@click.command(help="Process One")
@click.option("--inbound", type=click.STRING, default="data/inbound", help="inbound directory")
@click.option("--outbound", type=click.STRING, default="data/outbound", help="outbound directory")
@click.option("--manifest", type=click.STRING, help="File list json manifest")
@click.option("--run-name", type=click.STRING, default="workflow-step-process-one", help="The name of the run")
@click.option("--unique", type=click.BOOL, default=True, help="Flag for appending a nonce to the end of run names")
def run(inbound: str, outbound: str, manifest: str, run_name: str, unique: bool):
    warnings.filterwarnings("ignore")

    with mlflow.start_run(nested=True, run_name=build_run_name(run_name=run_name, unique=unique)):
        mlflow.log_param(key="inbound", value=inbound)
        mlflow.log_param(key="outbound", value=outbound)
        manifest_dict: Dict = json.loads(manifest)

        mlflow.log_dict(
            dictionary={"inbound": inbound, "outbound": outbound, "manifest": manifest_dict},
            artifact_file="business_metrics.json",
        )

        for file in manifest_dict["files"]:
            print(file)


if __name__ == "__main__":
    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
