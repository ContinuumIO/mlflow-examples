"""

"""

import uuid
import warnings
from pathlib import Path

import click
import keras_cv
import mlflow
from PIL import Image

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from ..utils.tracking import build_run_name, upsert_experiment


@click.option("--request-id", type=click.STRING, help="the request id")
@click.option("--data-base-dir", type=click.STRING, default="data", help="the base data directory that requests are stored in")
@click.option("--batch-size", type=click.INT, default=3, help="number of images to generate per batch")
@click.option("--image-width", type=click.INT, default=512, help="image width")
@click.option("--image-height", type=click.INT, default=512, help="image height")
@click.option("--run-name", type=click.STRING, default="workflow-step-process-data", help="The name of the run")
@click.option(
    "--unique", type=click.BOOL, default=True, help="Flag for appending a unique string to the end of run names"
)
@click.command(help="Workflow Step [Process Data]")
def run(request_id: str, data_base_dir: str, batch_size: int, image_width: int, image_height: int, run_name: str, unique: bool) -> None:
    """
    Runs the Workflow Step ['Worker' Process Data]
    """

    warnings.filterwarnings("ignore")

    with mlflow.start_run(nested=True, run_name=build_run_name(name=run_name, unique=unique)):
        mlflow.log_param(key="request_id", value=request_id)
        mlflow.log_param(key="data_base_dir", value=data_base_dir)
        mlflow.log_param(key="batch_size", value=batch_size)
        mlflow.log_param(key="image_width", value=image_width)
        mlflow.log_param(key="image_height", value=image_height)

        request_base: Path = Path(".") / data_base_dir / request_id

        request_output: Path = request_base / "output"
        request_output.mkdir(parents=True, exist_ok=True)

        prompt_file: str = (request_base/ "prompt.txt").as_posix()
        with open(file=prompt_file, mode="r", encoding="utf-8") as file:
            prompt: str = file.read()
        mlflow.log_text(text=prompt, artifact_file="prompt.txt")

        model = keras_cv.models.StableDiffusion(img_width=image_width, img_height=image_height)
        arrays = model.text_to_image(prompt, batch_size=batch_size)
        for array in arrays:
            image = Image.fromarray(array)
            output_file_path: str = (request_output / f"{str(uuid.uuid4())}.png").as_posix()
            image.save(output_file_path)


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
