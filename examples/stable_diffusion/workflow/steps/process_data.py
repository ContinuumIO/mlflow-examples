"""
Workflow Step [Process Data] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m workflow.steps.process_data`
When invoked this way the click defaults are used.

2. MLFlow CLI:
`mlflow run . -e process_data`
When invoked this way the MLproject default parameters are used

3. Workflow (or other code)
The function and its set up can be called from other code.
The `main` step does this in the workflow definition.

Note:
    If run stand alone (just the step) the run will report to a new job,
    rather than under a parent job (since one does not exist).
"""
import math
import time
import uuid
import warnings
from pathlib import Path
from typing import List, Optional

import click
import keras_cv
import mlflow
import numpy
from keras_cv.models.stable_diffusion.stable_diffusion import StableDiffusion
from PIL import Image

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from mlflow_adsp import create_unique_name, upsert_experiment


@click.command(help="Workflow Step [Process Data]")
@click.option("--request-id", type=click.STRING, help="The request ID.")
@click.option(
    "--data-base-dir", type=click.STRING, default="data", help="The base data directory that requests are stored in."
)
@click.option("--batch-size", type=click.INT, default=1, help="Number of images to generate per batch.")
@click.option("--num-steps", type=click.INT, default=50, help="The number of generation steps.")
@click.option("--image-width", type=click.INT, default=512, help="Image Width")
@click.option("--image-height", type=click.INT, default=512, help="Image Height")
@click.option(
    "--run-name",
    type=click.STRING,
    default="workflow-step-process-data",
    help="The base name of the run (for reporting to MLFlow).",
)
def run(
    request_id: str,
    data_base_dir: str,
    batch_size: int,
    num_steps: int,
    image_width: int,
    image_height: int,
    run_name: str,
) -> None:
    """
    Runs the Workflow Step ['Worker' Process Data]

    Parameters
    ----------
    request_id: str
        The request ID.
    data_base_dir: str
        Default: `data`
        The base data directory that requests are stored in.
    batch_size: int
        Default: 1
        Number of images to generate per batch.
    run_name: str
        The base name of the run (for reporting to MLFlow).
    image_width: int
        Default: 512
        Image Width
    image_height: int
        Default: 512
        Image Height
    num_steps: int:
        The number of generation steps.
    """

    warnings.filterwarnings("ignore")

    with mlflow.start_run(nested=True, run_name=create_unique_name(name=run_name)):
        seed: int = math.floor(time.time())

        mlflow.log_param(key="request_id", value=request_id)
        mlflow.log_param(key="data_base_dir", value=data_base_dir)
        mlflow.log_param(key="batch_size", value=batch_size)
        mlflow.log_param(key="image_width", value=image_width)
        mlflow.log_param(key="image_height", value=image_height)
        mlflow.log_param(key="num_steps", value=num_steps)
        mlflow.log_param(key="seed", value=seed)

        request_base: Path = Path(".") / data_base_dir / request_id

        request_output: Path = request_base / "output"
        request_output.mkdir(parents=True, exist_ok=True)

        prompt_file: str = (request_base / "prompt.txt").as_posix()
        with open(file=prompt_file, mode="r", encoding="utf-8") as file:
            prompt: str = file.read()
        mlflow.log_text(text=prompt, artifact_file="prompt.txt")

        model: StableDiffusion = keras_cv.models.StableDiffusion(
            img_width=image_width, img_height=image_height, jit_compile=True
        )
        arrays: List[numpy.ndarray] = model.text_to_image(prompt, batch_size=batch_size, num_steps=num_steps, seed=seed)
        for array in arrays:
            image: Image = Image.fromarray(array)
            filename: str = f"{str(uuid.uuid4())}.png"
            mlflow.log_image(image=image, artifact_file=filename)


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
