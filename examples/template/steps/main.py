import uuid

import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets


def run():
    with mlflow.start_run(run_name=f"parameterized-training-{str(uuid.uuid4())}", nested=True) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #
        training_data = "data/category/set/training.csv"
        experiment_id = run.info.experiment_id

        p = mlflow.projects.run(
            uri=".",
            entry_point="process_one",
            run_id=run.info.run_id,
            env_manager="local",
            backend="adsp",
            parameters={"training_data": training_data},
            experiment_id=experiment_id,
            synchronous=False,  # Allow the run to fail
        )
        succeeded = p.wait()
        p.get_log()


if __name__ == "__main__":
    load_ae5_user_secrets()
    run()
