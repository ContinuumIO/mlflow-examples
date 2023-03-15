# MLFlow Background Jobs On Anaconda Enterprise

## Overview
This single step workflow demonstrates the ability to run mlflow workflows Anaconda Enterprise Project Run-Now Jobs.

A `run-now` job will be created on the the project.  The name of the schedule will be the `run_id` of the child MLFlow run.

## Notebooks

* `entry_point` notebook has a stand-alone example of starting the `process_one` step of the workflow

## MLFlow Step Execution Through Command Line

* If you are executing these through the command line, then the environment variables MLFlow needs to communicate with the tracking server **MUST** be defined.
  * The easiest way to accomplish this is to uncomment the appropriate lines within anaconda-project.yml


Run the `main` entry point for the MLFlow multi-step workflow:
> anaconda-project run Main

Run the `process_one` step of the workflow:
> anaconda-project run ProcessOneStep
