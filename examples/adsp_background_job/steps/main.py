import uuid

import mlflow

from ..utils.environment_utils import init


def main():
    # Setup mlflow environment
    init()

    with mlflow.start_run(run_name=f"parameterized-training-{str(uuid.uuid4())}", nested=True) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #

        # Set up the job.
        training_data = "data/category/set/training.csv"
        experiment_id = run.info.experiment_id

        # Execute the workflow step.
        background_job = mlflow.projects.run(
            uri=".",
            entry_point="process_one",
            run_id=run.info.run_id,
            env_manager="local",
            backend="adsp",
            parameters={"training_data": training_data},
            experiment_id=experiment_id,
            synchronous=False,
        )

        # Wait for the job to complete.
        background_job = background_job.wait()

        # Get the log.
        background_job.get_log()


if __name__ == "__main__":
    main()
