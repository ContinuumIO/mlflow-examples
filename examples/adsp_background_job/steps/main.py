import uuid

import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from mlflow_adsp import upsert_experiment

def run():
    with mlflow.start_run(run_name=f"parameterized-training-{str(uuid.uuid4())}", nested=True) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #

        # Set up the job.
        training_data = "data/category/set/training.csv"
        experiment_id = run.info.experiment_id

<<<<<<<< HEAD:examples/template/steps/main.py
        p = mlflow.projects.run(
========
        # Execute the workflow step.
        background_job = mlflow.projects.run(
>>>>>>>> 123f268f75082fa742e28796bb8d27dcf6738151:examples/adsp_background_job/steps/main.py
            uri=".",
            entry_point="process_one",
            run_id=run.info.run_id,
            env_manager="local",
            backend="adsp",
            parameters={"training_data": training_data},
            experiment_id=experiment_id,
            synchronous=False,  # Allow the run to fail
        )
<<<<<<<< HEAD:examples/template/steps/main.py
        succeeded = p.wait()
        p.get_log()
========

        # Wait for the job to complete.
        background_job = background_job.wait()

        # Get the log.
        background_job.get_log()
>>>>>>>> 123f268f75082fa742e28796bb8d27dcf6738151:examples/adsp_background_job/steps/main.py


if __name__ == "__main__":
    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
