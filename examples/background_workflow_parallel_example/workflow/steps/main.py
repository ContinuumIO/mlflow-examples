"""
Workflow Step [Main] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m steps.main`
When invoked this way the click defaults are used.

2. MLFlow CLI:
`mlflow run . --backend local`
- or -
`mlflow run . --backend adsp`
When invoked this way the MLproject default parameters are used

3. Anaconda-Project Commands
`anaconda-project run workflow:main:local`
- or -
`anaconda-project run workflow:main:adsp`
"""

import json
import math
from pathlib import Path
from typing import Dict, List

import click
import mlflow
from mlflow.projects.submitted_run import LocalSubmittedRun
from mlflow_adsp import ADSPSubmittedRun, ExecuteStepRequest, execute_step, process_work_queue

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from ..utils.tracking import build_run_name, upsert_experiment
from ..utils.worker import get_batches


@click.command(help="Workflow [Main]")
@click.option("--work-dir", type=click.STRING, default="data", help="The base directory to work within")
@click.option("--inbound", type=click.STRING, default="inbound", help="The inbound directory")
@click.option("--outbound", type=click.STRING, default="outbound", help="The outbound directory")
@click.option(
    "--batch-size", type=click.IntRange(min=1, max=100), default=1, help="Batch size (as percentage) for each worker"
)
@click.option("--run-name", type=click.STRING, default="parallel-data-processing-job", help="The name of the run")
@click.option(
    "--unique", type=click.BOOL, default=True, help="Flag for appending a unique string to the end of run names"
)
@click.option("--backend", type=click.STRING, default="local", help="Backend to use")
# pylint: disable=too-many-locals
def workflow(
        work_dir: str, inbound: str, outbound: str, batch_size: int, run_name: str, unique: bool, backend: str
) -> None:
    """

    Parameters
    ----------
    work_dir: str
        The base directory to work within
    inbound: str
        The inbound directory
    outbound: str
        The outbound directory
    batch_size: int
        Batch size (as percentage) for each worker
    run_name: str
        The name of the run
    unique: bool
        Flag for appending a unique string to the end of run names
    backend: str
        The backend to use for workers.
    """

    with mlflow.start_run(run_name=build_run_name(name=run_name, unique=unique)) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #

        #############################################################################
        # Set up runtime environment
        #############################################################################

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

        #############################################################################
        # Execute workflow steps
        #############################################################################

        #############################################################################
        # Download Step
        #############################################################################
        execute_step(
            request=ExecuteStepRequest(
                entry_point="download_real_esrgan",
                parameters={"source_dir": source_path},
                run_name=build_run_name(
                    name="workflow-step-download-real-esrgan",
                    unique=unique
                ),
                synchronous=True,
                backend="local"
            )
        )

        #############################################################################
        # Prepare Worker Environment Step
        #############################################################################
        execute_step(
            request=ExecuteStepRequest(
                entry_point="prepare_worker_environment",
                parameters={"backend": backend},
                run_name=build_run_name(
                    name="workflow-step-prepare-worker-environment",
                    unique=unique
                ),
                synchronous=True,
                backend="local"
            )
        )

        #############################################################################
        # Processing Step [Parallel]
        #############################################################################
        file_count: int = len(file_list)
        if file_count > 0:
            batch_amount: int = math.floor(file_count * (batch_size / 100))
            batch_amount = batch_amount if batch_amount > 0 else 1
            batches: List = get_batches(batch_size=batch_amount, source_list=file_list)

            print(f"batch size: {batch_size}")
            print(f"batch amount: {batch_amount}")
            print(f"number of batches: {len(batches)}")

            print("starting workers")
            jobs: List[ExecuteStepRequest] = []
            for batch in batches:
                process_manifest: Dict = {"files": batch}

                request: ExecuteStepRequest = ExecuteStepRequest(
                    entry_point="process_data",
                    parameters={
                        "inbound": inbound_path.as_posix(),
                        "outbound": outbound_path.as_posix(),
                        "manifest": json.dumps(process_manifest),
                    },
                    run_name=build_run_name(name="workflow-step-process-data", unique=unique),
                    backend=backend,
                    backend_config={
                        "resource_profile": "large"
                    },
                    synchronous=False,
                )
                jobs.append(request)

            # submit jobs
            submitted_runs: List[ADSPSubmittedRun] = process_work_queue(jobs=jobs)

            for result in [run for run in submitted_runs if type(run) != LocalSubmittedRun]:
                run_log = result.get_log()
                print(run_log)

        else:
            print("No files in `inbound` found to process, skipping step")


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    workflow()
