# MLFlow Workflow Step With Anaconda Enterprise Background Job 

## Overview
This single step workflow demonstrates the ability to run mlflow workflow steps as Anaconda Enterprise Project Jobs.

A `run-now` job will be created and executed on the project.  The name of the schedule will be the `run_id` of the child MLFlow run.

## Notebooks

* `entry_point` notebook has a stand-alone example of starting the `process_one` step of the workflow

## MLFlow Step Execution Through Command Line

* If you are executing these through the command line, then the environment variables MLFlow needs to communicate with the tracking server **MUST** be defined.
* The easiest way to accomplish this is to uncomment the appropriate lines within `anaconda-project.yml`


Run the `main` entry point for the MLFlow multistep workflow:
```commandline
anaconda-project run Main
```

Run the `process_one` step of the workflow:
```commandline
anaconda-project run ProcessOneStep
```
