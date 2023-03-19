import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import click
import mlflow
from mlflow.projects.submitted_run import LocalSubmittedRun, SubmittedRun

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from .utils import build_run_name, upsert_experiment, get_batches


def execute_step(
    entry_point: str,
    parameters: dict,
    run_id: Optional[str] = None,
    backend: Optional[str] = "local",
    synchronous: Optional[bool] = False,
    run_name: Optional[str] = None,
) -> SubmittedRun:
    launch_parameters: Dict = {
        "uri": ".",
        "entry_point": entry_point,
        "parameters": parameters,
        "env_manager": "local",
        "synchronous": synchronous,
    }
    if run_id:
        launch_parameters["run_id"] = run_id
    if backend:
        launch_parameters["backend"] = backend
    if run_name:
        launch_parameters["run_name"] = run_name

    print(f"Launching new background job for entrypoint={entry_point} and parameters={launch_parameters}")
    return mlflow.projects.run(**launch_parameters)


@click.command(help="Workflow [Main]")
@click.option("--work-dir", type=click.STRING, default="data", help="The base directory to work within")
@click.option("--inbound", type=click.STRING, default="inbound", help="inbound directory")
@click.option("--outbound", type=click.STRING, default="outbound", help="outbound directory")
@click.option(
    "--batch-size", type=click.IntRange(min=1, max=100), default=1, help="batch size (as percentage) for each worker"
)
@click.option("--run-name", type=click.STRING, default="jburt-parameterized-job", help="The name of the run")
@click.option("--unique", type=click.BOOL, default=True, help="Flag for appending a nonce to the end of run names")
@click.option("--backend", type=click.STRING, default="local", help="Backend to use")
def workflow(work_dir: str, inbound: str, outbound: str, batch_size: int, run_name: str, unique: bool, backend: str):
    with mlflow.start_run(run_name=build_run_name(run_name=run_name, unique=unique), nested=True) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #

        # Set up runtime environment

        print(f"work dir={work_dir}")
        print(f"inbound={inbound}")
        print(f"outbound={outbound}")
        print(f"batch size={batch_size}")

        run_id: str = run.info.run_id
        print(f"run_id: {run_id}")

        # Resolve paths
        base_path: Path = Path(work_dir)
        inbound_path: Path = base_path / "inbound"
        outbound_path: Path = base_path / "outbound"
        source_path: Path = base_path / "Real-ESRGAN"

        #  Ensure a sane runtime environment
        inbound_path.mkdir(parents=True, exist_ok=True)
        outbound_path.mkdir(parents=True, exist_ok=True)

        # Generate file list to process
        file_list: List[str] = []
        for item in inbound_path.glob("*"):
            if item.is_file():
                file_list.append(item.name)

        # Execute workflow steps

        # Download Step
        download_step: Union[SubmittedRun, LocalSubmittedRun] = execute_step(
            entry_point="download_real_esrgan",
            parameters={"source_dir": source_path},
            run_name=build_run_name(run_name="workflow-step-download-real-esrgan", unique=unique),
        )
        download_step.wait()

        # This is only relevant when we start using the adsp backend
        # if isinstance(download_step, SubmittedRun):
        #     download_step.get_log()


        # Prepare Worker Environment Step
        download_step: Union[SubmittedRun, LocalSubmittedRun] = execute_step(
            entry_point="prepare_worker_environment",
            parameters={"backend": backend},
            run_name=build_run_name(run_name="workflow-step-prepare-worker-environment", unique=unique)
        )
        download_step.wait()

        # Processing Step [Parallel]
        file_count: int = len(file_list)
        if file_count > 0:
            batch_amount: int = math.floor(file_count * (batch_size / 100))
            batch_amount = batch_amount if batch_amount > 0 else 1
            batches: List = get_batches(batch_size=batch_amount, file_list=file_list)

            print(f"batch size: {batch_size}")
            print(f"batch amount: {batch_amount}")
            print(f"number of batches: {len(batches)}")

            for batch in batches:
                process_manifest: Dict = {"files": batch}

                # There is a single step (Process One)
                background_job: Union[SubmittedRun, LocalSubmittedRun, Any] = execute_step(
                    entry_point="process_one",
                    parameters={
                        "inbound": inbound_path.as_posix(),
                        "outbound": outbound_path.as_posix(),
                        "manifest": json.dumps(process_manifest),
                    },
                    run_name=build_run_name(run_name="workflow-step-process-one", unique=unique),
                    backend=backend
                )

                background_job.wait()

                # This is only relevant when we start using the adsp backend
                # if isinstance(download_step, SubmittedRun):
                #     download_step.get_log()
        else:
            print("No files in `inbound` found to process, skipping step")

if __name__ == "__main__":
    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    workflow()
