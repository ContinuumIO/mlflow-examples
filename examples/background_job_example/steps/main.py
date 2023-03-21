import uuid

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from .utils import upsert_experiment, build_run_name


@click.command(help="Workflow [Main]")
@click.option("--run-name", type=click.STRING, default="parameterized-job", help="The name of the run")
@click.option(
    "--unique", type=click.BOOL, default=True, help="Flag for appending a unique string to the end of run names"
)
@click.option("--training-data", type=click.STRING, default="data/category/set/training.csv", help="The training data")
def workflow(run_name: str, unique: bool, training_data: str):
    with mlflow.start_run(run_name=build_run_name(run_name=run_name, unique=unique), nested=True) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #
        training_data = "data/category/set/training.csv"
        experiment_id = run.info.experiment_id

        background_job = mlflow.projects.run(
            uri=".",
            entry_point="process_one",
            run_id=run.info.run_id,
            env_manager="local",
            backend="adsp",
            parameters={"training_data": training_data},
            experiment_id=experiment_id,
            synchronous=False
        )
        background_job = background_job.wait()
        background_job.get_log()


if __name__ == "__main__":
    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    workflow()
