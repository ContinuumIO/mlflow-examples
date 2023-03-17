import uuid

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets


def execute_step(entry_point: str, parameters: dict, run_id: str):
    print(f"Launching new background job for entrypoint={entry_point} and parameters={parameters}")
    return mlflow.projects.run(
        uri=".",
        entry_point=entry_point,
        parameters=parameters,
        env_manager="local",
        backend="adsp",
        run_id=run_id,
        synchronous=False,  # we defer control of this to the caller
    )


@click.command(help="Workflow [Main]")
@click.option("--inbound", type=click.STRING, default="data/inbound", help="inbound directory")
@click.option("--outbound", type=click.STRING, default="data/outbound", help="outbound directory")
def workflow(inbound: str, outbound: str):
    with mlflow.start_run(run_name=f"jburt-parameterized-job-{str(uuid.uuid4())}", nested=True) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #

        print(f"inbound={inbound}")
        print(f"outbound={outbound}")

        run_id: str = run.info.run_id
        print(f"run_id: {run_id}")

        # There is a single step (Process One)
        background_job = execute_step(
            entry_point="process_one",
            parameters={"inbound": inbound, "outbound": outbound},
            run_id=run_id,
        )

        background_job.wait()
        background_job.get_log()


if __name__ == "__main__":
    load_ae5_user_secrets()
    workflow()
