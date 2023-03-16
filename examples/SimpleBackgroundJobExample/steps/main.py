import uuid

import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets


def execute_step(entry_point: str, parameters: dict, experiment_id: str, run_id: str):
    print(f"Launching new background job for entrypoint={entry_point} and parameters={parameters}")
    return mlflow.projects.run(
        uri=".",
        entry_point=entry_point,
        parameters=parameters,
        env_manager="local",
        backend="adsp",
        experiment_id=experiment_id,
        run_id=run_id,
        synchronous=False,
    )


def workflow():
    experiment_name: str = "SimpleBackgroundJobExample"

    try:
        experiment = mlflow.get_experiment_by_name(name=experiment_name)
        experiment_id: str = experiment.experiment_id
    except Exception as error:
        experiment_id: str = mlflow.create_experiment(name=experiment_name)

    with mlflow.start_run(run_name=f"parameterized-training-{str(uuid.uuid4())}") as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #
        training_data = "data/category/set/training.csv"
        run_id = run.info.run_id

        background_job = execute_step(
            entry_point="process_one",
            parameters={"training_data": training_data},
            experiment_id=experiment_id,
            run_id=run_id,
        )

        background_job.wait()
        background_job.get_log()


if __name__ == "__main__":
    load_ae5_user_secrets()
    workflow()
