import warnings

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets


# Note: If run stand alone (just the step) the run will report to a new job,
# rather than under a parent job (since one does not exist).
@click.command(help="Process One")
@click.option("--inbound", type=click.STRING, default="data/inbound", help="inbound directory")
@click.option("--outbound", type=click.STRING, default="data/outbound", help="outbound directory")
def run(inbound, outbound):
    warnings.filterwarnings("ignore")

    with mlflow.start_run(nested=True):
        mlflow.log_param(key="inbound", value=inbound)
        mlflow.log_param(key="outbound", value=outbound)

        mlflow.log_dict(dictionary={"inbound": inbound, "outbound": outbound}, artifact_file="business_metrics.json")


if __name__ == "__main__":
    load_ae5_user_secrets()
    run()
