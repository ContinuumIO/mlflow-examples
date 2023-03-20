"""
Workflow Step ['Worker' Process Data] Definition

This step can be invoked in three different ways:
1. Python module invocation:
`python -m steps.process_data`
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

import json
import warnings
from pathlib import Path
from typing import Dict

import click
import mlflow

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets

from .utils import build_run_name, process_launch_wait, upsert_experiment


@click.command(help="Workflow Step ['Worker' Process Data]")
@click.option("--inbound", type=click.STRING, default="data/inbound", help="inbound directory")
@click.option("--outbound", type=click.STRING, default="data/outbound", help="outbound directory")
@click.option(
    "--source-dir", type=click.STRING, default="data/Real-ESRGAN", help="The source directory for real-esrgran"
)
@click.option("--manifest", type=click.STRING, help="File list json manifest")
@click.option("--run-name", type=click.STRING, default="workflow-step-process-data", help="The name of the run")
@click.option("--unique", type=click.BOOL, default=True, help="Flag for appending a nonce to the end of run names")
@click.option("--force", type=click.BOOL, default=False, help="Flag for over-riding output files if they exist")
def run(inbound: str, outbound: str, source_dir: str, manifest: str, run_name: str, unique: bool, force: bool):
    warnings.filterwarnings("ignore")

    with mlflow.start_run(nested=True, run_name=build_run_name(run_name=run_name, unique=unique)):
        mlflow.log_param(key="inbound", value=inbound)
        mlflow.log_param(key="outbound", value=outbound)
        manifest_dict: Dict = json.loads(manifest)

        mlflow.log_dict(
            dictionary={"inbound": inbound, "outbound": outbound, "manifest": manifest_dict},
            artifact_file="business_metrics.json",
        )

        for file in manifest_dict["files"]:
            inbound_file: Path = Path(inbound) / file
            outbound_file: Path = Path(outbound) / (Path(file).stem + "_out" + Path(file).suffix)
            outbound_file_path: Path = Path(outbound)

            if outbound_file.exists() and not force:
                print(f"Skipping {file}, it already exists in the outbound folder")
                break

            if outbound_file.exists() and force:
                outbound_file.unlink(missing_ok=True)

            cmd: str = (
                "python -m inference_realesrgan "
                f"--input {inbound_file.resolve()} "
                f"--output {outbound_file_path.resolve()} "
                "--model_path ../weights/RealESRGAN_x4plus.pth "
                "--fp32 "
            )
            print(cmd)
            process_launch_wait(shell_out_cmd=cmd, cwd=source_dir)


if __name__ == "__main__":
    # Ensure:
    #  1. We load AE5 secrets
    #  2. That we have set our experiment name for reporting.
    #     See notes in anaconda-project.xml around MLFlow project naming control.

    load_ae5_user_secrets()
    mlflow.set_experiment(experiment_id=upsert_experiment())
    run()
