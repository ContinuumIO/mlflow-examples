"""
Workflow Step [Main] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m workflow.steps.main`
When invoked this way the click defaults are used.

2. MLFlow CLI:
`mlflow run . --backend local`
- or -
`mlflow run . --backend adsp`
When invoked this way the MLproject default parameters are used.

3. Anaconda-Project Commands
`anaconda-project run workflow:main:local`
- or -
`anaconda-project run workflow:main:adsp`
"""

import math
import uuid
from pathlib import Path
from typing import List

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from mlflow_adsp import Job, Scheduler, Step, create_unique_name, upsert_experiment

# To see debug level output of scheduler un-comment the below.
# import logging
# logging.basicConfig()
# logging.getLogger("mlflow_adsp.common.scheduler").setLevel(level=logging.DEBUG)


@click.command(help="Workflow [Main]")
@click.option("--prompt", type=click.STRING, help="The prompt to use for image generation.")
@click.option(
    "--data-base-dir", type=click.STRING, default="data", help="The base data directory that requests are stored in."
)
@click.option("--total-batch-size", type=click.INT, default=9, help="Number of total images to generate.")
@click.option(
    "--per-worker-batch-size", type=click.INT, default=1, help="Number of images to generate per worker invocation."
)
@click.option("--num-steps", type=click.INT, default=50, help="The number of generation steps.")
@click.option("--image-width", type=click.INT, default=512, help="Image Width")
@click.option("--image-height", type=click.INT, default=512, help="Image Height")
@click.option(
    "--run-name", type=click.STRING, default="workflow-stable-diffusion-parallel", help="The name of the run."
)
@click.option("--backend", type=click.STRING, default="local", help="The backend to use for workers.")
def workflow(
    prompt: str,
    data_base_dir: str,
    total_batch_size: int,
    per_worker_batch_size: int,
    num_steps: int,
    image_width: int,
    image_height: int,
    run_name: str,
    backend: str,
) -> None:
    """
    Workflow Entry Point

    Parameters
    ----------
    prompt: str
        The prompt to use for image generation.
    data_base_dir: str
        Default: `data`
        The base data directory that requests are stored in.
    total_batch_size: int
        Default: 9
        Number of total images to generate.
    per_worker_batch_size: int
        Default: 1
        Number of images to generate per worker invocation.
    num_steps: int:
        The number of generation steps.
    image_width: int
        Default: 512
        Image Width
    image_height: int
        Default: 512
        Image Height
    run_name: str
        Default: `workflow-step-process-data`
        The name of the run.
    backend: str
        Default: `local`
        The backend to use for workers.
    """

    with mlflow.start_run(run_name=create_unique_name(name=run_name)) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #

        #############################################################################
        # Set up runtime environment
        #############################################################################

        print(f"prompt={prompt}")
        print(f"data_base_dir={data_base_dir}")
        print(f"total_batch_size={total_batch_size}")
        print(f"per_worker_batch_size={per_worker_batch_size}")
        print(f"num_steps: {num_steps}")
        print(f"image_width={image_width}")
        print(f"image_height={image_height}")
        print(f"backend={backend}")

        run_id: str = run.info.run_id
        print(f"run_id: {run_id}")

        request_id: str = str(uuid.uuid4())
        base_path: Path = Path(data_base_dir) / request_id
        base_path.mkdir(parents=True, exist_ok=True)
        with open(file=(base_path / "prompt.txt").as_posix(), mode="w", encoding="utf-8") as file:
            file.write(prompt)

        #############################################################################
        # Execute workflow steps
        #############################################################################

        #############################################################################
        # Prepare Worker Environment Step
        #############################################################################
        Scheduler.execute_step(
            step=Step(
                entry_point="prepare_worker_environment",
                parameters={"backend": backend},
                run_name=create_unique_name(name="workflow-step-prepare-worker-environment"),
                synchronous=True,
                backend="local",
            )
        )

        #############################################################################
        # Processing Step
        #############################################################################
        worker_count: int = math.ceil(total_batch_size / per_worker_batch_size)
        print(f"number of workers: {worker_count}")

        # build requests

        steps: List[Step] = []
        for _ in range(worker_count):
            step: Step = Step(
                entry_point="process_data",
                parameters={
                    "request_id": request_id,
                    "data_base_dir": data_base_dir,
                    "batch_size": per_worker_batch_size,
                    "image_width": image_width,
                    "image_height": image_height,
                    "num_steps": num_steps,
                },
                run_name=create_unique_name(name="workflow-step-process-data"),
                backend=backend,
                backend_config={"resource_profile": "large"},
                synchronous=backend == "local",  # Force to serial processing if running locally.
            )
            steps.append(step)

        # submit steps
        print("starting workers")
        adsp_jobs: List[Job] = Scheduler().process_work_queue(steps=steps)

        print("Step execution completed")
        for job in adsp_jobs:
            print(f"Job ID: {job.id}, Status: {job.last_seen_status}, Number of executions: {len(job.runs)}")


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    workflow()
