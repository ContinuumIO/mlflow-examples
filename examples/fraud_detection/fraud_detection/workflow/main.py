"""
Workflow Step [Main] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m workflow.steps.main`
When invoked this way the click defaults are used.

2. Anaconda Project Commands
`anaconda-project run workflow:main:local`
- or -
`anaconda-project run workflow:main:adsp`
"""

import logging

import click
import mlflow
from mlflow_adsp import Job, Scheduler, Step, create_unique_name

from fraud_detection.services.mlflow_helper import mlflow_environment_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments
@click.command(help="Workflow [Main]")
@click.option(
    "--run-name",
    type=click.STRING,
    default="workflow-fraud-detection-model-training-parallel",
    help="The name of the run.",
)
@click.option("--backend", type=click.STRING, default="local", help="The backend to use for workers.")
@click.option("--model-type", type=click.STRING, multiple=True)
@click.option("--strategy", type=click.STRING, multiple=True)
@click.option(
    "--cv-n-reps",
    type=click.INT,
    default=10,
)
@click.option(
    "--cv-n-splits",
    type=click.INT,
    default=5,
)
@click.option(
    "--resource-profile",
    type=click.STRING,
    default="default",
)
def main(
    run_name: str,
    backend: str,
    model_type: list[str],
    strategy: list[str],
    cv_n_reps: int = 10,
    cv_n_splits: int = 5,
    resource_profile: str = "default",
) -> None:
    """
    Workflow Entry Point

    Parameters
    ----------
    run_name: str
        The MLflow run name.
    backend: str
        Default: `local`
        The backend to use for workers.
    model_type: list[str]
        The list of model types to train. (Supported by the factory)
    strategy: list[str]
        The supported sampling strategies to use during training. (Supported by the factory)
    cv_n_reps: int
        Number of times cross-validator needs to be repeated.
    cv_n_splits: int
        Number of folds. Must be at least 2.
    resource_profile: str = "default"
        The resource profile to run the training on.
    """

    # Init our MLflow experiment environment
    mlflow_environment_config()

    with mlflow.start_run(run_name=create_unique_name(name=run_name)) as run:
        #
        # Wrapped and Tracked Workflow Step Runs
        # https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run
        #

        #############################################################################
        # Set up runtime environment
        #############################################################################

        logger.info("Backend: %s", backend)

        run_id: str = run.info.run_id
        logger.info("Run ID: %s", run_id)

        #############################################################################
        # Execute workflow steps
        #############################################################################

        #############################################################################
        # Prepare Worker Environment Step
        #############################################################################
        work_step: Step = Step(
            entry_point="prepare_worker_environment",
            parameters={"backend": backend},
            run_name=create_unique_name(name="prepare-worker-environment"),
            synchronous=True,
            backend="local",
        )
        Scheduler.execute_step(step=work_step)

        #############################################################################
        # Processing Step
        #############################################################################

        # build steps
        steps: list[Step] = []

        for local_type in model_type:
            for local_strategy in strategy:
                step: Step = Step(
                    entry_point="train",
                    parameters={
                        "run_name": f"workflow-step-train-{local_type}-{local_strategy}",
                        "model_type": local_type,
                        "strategy": local_strategy,
                        "cv_n_reps": cv_n_reps,
                        "cv_n_splits": cv_n_splits,
                    },
                    run_name=create_unique_name(name=f"workflow-step-train-{local_type}-{local_strategy}"),
                    backend=backend,
                    backend_config={"resource_profile": resource_profile},
                    synchronous=backend == "local",  # Force to serial processing if running locally.
                )
                steps.append(step)

        logger.info(steps)

        # submit steps
        logger.info("starting workers")
        adsp_jobs: list[Job] = Scheduler().process_work_queue(steps=steps)

        logger.info("Step execution completed")
        for job in adsp_jobs:
            logger.info("Job ID: %s, Status: %s, Number of executions: %i", job.id, job.last_status, len(job.runs))

        #############################################################################
        # Champion / Challenger Analysis
        #############################################################################

        Scheduler.execute_step(
            step=Step(
                entry_point="champion_challenger_promotion",
                parameters={"run_name": "champion-challenger-promotion", "parent_run_id": run_id},
                run_name=create_unique_name(name="champion-challenger-promotion"),
                synchronous=True,
                backend=backend,
            )
        )


if __name__ == "__main__":
    main()
