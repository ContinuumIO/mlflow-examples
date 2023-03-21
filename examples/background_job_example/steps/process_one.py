import warnings

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from steps.utils import build_run_name, upsert_experiment


# Note: If run stand alone (just the step) the run will report to a new job,
# rather than under a parent job (since one does not exist).
@click.command(help="Workflow [Process One]")
@click.option("--some-parameter-int", type=click.INT, default=1, help="The integer for one")
@click.option("--some-parameter-float", type=click.FLOAT, default=1.0, help="The float for one")
@click.option("--some-parameter-string", type=click.STRING, default="1", help="The string for one")
@click.option("--training-data", type=click.STRING, default="data/category/set/training.csv", help="The training data")
@click.option("--run-name", type=click.STRING, default="background-job", help="The name of the run")
@click.option(
    "--unique", type=click.BOOL, default=True, help="Flag for appending a unique string to the end of run names"
)
def run(some_parameter_int: int, some_parameter_float: float, some_parameter_string: str, training_data: str, run_name: str, unique: bool):
    warnings.filterwarnings("ignore")

    with mlflow.start_run(run_name=build_run_name(run_name=run_name, unique=unique), nested=True):
        mlflow.log_param(key="some_parameter_int", value=some_parameter_int)
        mlflow.log_param(key="some_parameter_float", value=some_parameter_float)
        mlflow.log_param(key="some_parameter_string", value=some_parameter_string)
        mlflow.log_param(key="training_data", value=training_data)

        mlflow.log_dict(dictionary={"sample_key": "sample_value"}, artifact_file="business_metrics.json")


if __name__ == "__main__":
    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
