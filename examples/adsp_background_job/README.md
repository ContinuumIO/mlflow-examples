# MLFlow Driven Background Jobs Within ADSP

## Overview
Demonstrates the ability to run MLFlow workflow steps as Anaconda Data Science Platform `Run-Now` Jobs.

This example does not leverage conda environment caching for the worker instances.
Other examples (including real_esrgran, and stable_diffusion) however do demonstrate multistep workflows with environment caching.

A `Run-Now` job will be created and executed on the project.  The name of the schedule will be the `run_id` of the child MLFlow run.

## Setup
1. Download the solution.
2. Ensure the variable `MLFLOW_EXPERIMENT_NAME` within the `anaconda-project.yml` is updated appropriately.
3. Upload the project into AE5
> ae5 project upload .
4. Start a project session and allow conda to complete dependency installation.
5. Ensure you have the below AE5 secrets defined, or uncommented and added to the `anaconda-project.yml` file.
    
    | Variable              |
    |-----------------------|
    | AE5_HOSTNAME          |
    | AE5_USERNAME          |
    | AE5_PASSWORD          |
    | MLFLOW_TRACKING_URI   |
    | MLFLOW_REGISTRY_URI   |
    | MLFLOW_TRACKING_TOKEN |

## Notebook

* `example` notebook has a stand-alone example of starting the `process_one` step of the workflow.

### Usage

Run the `main` entry point for the MLFlow multistep workflow:
```commandline
anaconda-project run Main
```

Run the `process_one` step of the workflow:
```commandline
anaconda-project run ProcessOneStep
```
