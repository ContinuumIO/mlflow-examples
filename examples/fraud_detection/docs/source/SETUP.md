# Setup

## Pre-requisite

1. Anaconda Enterprise 5.7+
2. MLflow has been deployed within AE5 as per [Adding MLflow to Anaconda Enterprise](https://enterprise-docs.anaconda.com/en/latest/admin/advanced/mlflow_install.html)

## Project Setup

1. Download the solution.
2. Upload the project to ADSP.
    > ae5 project upload .
3. Start a project session
4. Upload the data into `data`.
5. Allow conda to complete dependency installation. 
6. Ensure you have the ADSP Configuration Variables / Secrets AE5 Secrets defined, or uncommented and added to the `anaconda-project.yml` file.
See the below section `ADSP Configuration Variables / Secrets` for additional details.

7. Update domain name for the ADSP instance.
* These variables **MUST** be updated for the ADSP cluster running the endpoints.

    | Variable                    |
    |-----------------------------|
    | SELF_HOSTED_MODEL_ENDPOINT  |
    | SELF_HOSTED_DATA_STREAM     |
    | BOKEH_ALLOW_WS_ORIGIN       |

8. **[Optional]** Conda Environment Setup for parallel processing within ADSP

    This is not strictly required to run before the first training, however, its good check of the environment.
    > anaconda-project run bootstrap
    
    Run this command within the `default` conda environment created for the project.
    This command will create a copy usable by the background jobs.
    This prevents the processes from having to recreate the environment each time they run.
    If dependencies change, this environment would need to be updated.

9. Create scheduled training job
    * See [Scheduling deployments](https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/deployments/schedule-deploy.html#scheduling-deployments) for details on how to create a schedule job on AE5.
    * The project command to use is called `workflow:main:adsp`.
    * Sane schedules might be every three hours: `0 */3 * * *` or once daily `0 0 * * *`.
10. Start the Prediction API
    * See [Deploying a project](https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/deployments/index.html#deploying-a-project) for details on how to create deployments in AE5.
    * The project command to use is called `prediction-api`.
    * The deployment should be able to use a very small resource profile.
    * Set the URL to the value in the variable `SELF_HOSTED_MODEL_ENDPOINT`.
    * For the demo the endpoint can be made `public` and shared with `everyone`, however, if you choose to deploy as `private`:
      * Ensure you generate a private access token and store it as an AE5 Secret called `SELF_HOSTED_MODEL_ENDPOINT_TOKEN` on all consumer user accounts.
11. Start the Data Stream API
    * See [Deploying a project](https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/deployments/index.html#deploying-a-project) for details on how to create deployments in AE5.
    * The project command to use is called `dashboard`.
    * The deployment should be able to use a very small resource profile.
    * Set the URL to the value in the variable `SELF_HOSTED_DATA_STREAM`.
    * For the demo the endpoint can be made `public` and shared with `everyone`, however, if you choose to deploy as `private`:
      * Ensure you generate a private access token and store it as an AE5 Secret called `SELF_HOSTED_DATA_STREAM_TOKEN` on all consumer user accounts.
12. Start the Dashboard

    The Dashboard will consume the APIs started in steps 10 & 11 in addition to the MLflow tracking server.

    * See [Deploying a project](https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/deployments/index.html#deploying-a-project) for details on how to create deployments in AE5.
    * The project command to use is called `data-stream`.
    * The deployment should be able to use a very small resource profile.
    * Set the URL to the value in the variable `BOKEH_ALLOW_WS_ORIGIN`.
    * For the demo the endpoint can be made `public` and shared with `everyone`.
