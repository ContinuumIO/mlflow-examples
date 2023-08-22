import click
import mlflow

from ..utils.environment_utils import init


# Note: If run stand alone (just the step) the run will report to a new job,
# rather than under a parent job (since one does not exist).
@click.command(help="Process One")
@click.option("--some-parameter-int", type=click.INT, default=1, help="The integer for one")
@click.option("--some-parameter-float", type=click.FLOAT, default=1.0, help="The float for one")
@click.option("--some-parameter-string", type=click.STRING, default="1", help="The string for one")
@click.argument("training_data")
def process_one(training_data, some_parameter_int, some_parameter_float, some_parameter_string):
    init()

    with mlflow.start_run(nested=True):
        mlflow.log_param(key="training_data", value=training_data)
        mlflow.log_param(key="some_parameter_int", value=some_parameter_int)
        mlflow.log_param(key="some_parameter_float", value=some_parameter_float)
        mlflow.log_param(key="some_parameter_string", value=some_parameter_string)

        mlflow.log_dict(dictionary={"sample_key": "sample_value"}, artifact_file="business_metrics.json")


if __name__ == "__main__":
    process_one()
