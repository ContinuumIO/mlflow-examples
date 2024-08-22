# Variables

### Notebook

* For stand-alone notebooks.  Leave as-is under most conditions.

| Name                | Description                                                   | Default Value                                                |
|---------------------|:--------------------------------------------------------------|--------------------------------------------------------------|
| DATA                | Input data path                                               | data/creditcard.csv                                          |
| MODELS_BASE_DIR     | Location to store models in                                   | data/models                                                  |
| DECISION_TREE_SMOTE | joblib file path for decision tree + over sampling smote      | data/models/gs_decision_tree_over_sampling_smote.joblib      |
| DECISION_TREE_NMISS | joblib file path for decision tree + under sampling near miss | data/models/gs_decision_tree_under_sampling_near_miss.joblib |
| RANDOM_FOREST_SMOTE | joblib file path for random forest + over sampling smote      | data/models/gs_random_forest_over_sampling_smote.joblib      |
| RANDOM_FOREST_NMISS | joblib file path for random forest + under sampling near miss | data/models/gs_random_forest_under_sampling_near_miss.joblib |


### ADSP Configuration Variables / Secrets
* These **SHOULD** be created as AE5 User Secrets rather than as project environment variables.
* For additional details see [Storing Secrets](https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/user-settings.html?highlight=secrets#storing-secrets).

| Name                                     | Description                                                                                                                 |
|------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|
| MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING | Disable printing a warning when –env-manager=conda is specified                                                             |
| MLFLOW_TRACKING_INSECURE_TLS             | Allow insecure TLS (opens up man-in-the-middle attacks). Useful for development, do **NOT** enable in a production setting  |
| MLFLOW_REGISTRY_URI                      | The MLflow Registry URI                                                                                                     |
| MLFLOW_TRACKING_URI                      | The MLflow Tracking URI                                                                                                     |
| MLFLOW_TRACKING_TOKEN                    | API level access to MLflow will require a private access token. This is normally generated and provided by the Admin        |
| AE5_HOSTNAME                             | AE5 cluster F.Q.D.N.                                                                                                        |
| AE5_USERNAME                             | User account to launch training. This user must have permissions to the resource profile specified in the training workflow |
| AE5_PASSWORD                             | User password to launch training                                                                                            |

### Project Configuration

| Name                             | Default                                                       | Description                                                                                                       |
|----------------------------------|:--------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| PYTHONWARNINGS                   | ignore                                                        | Control for python `warning`.  Training can provide a large volume of warning messages.  These are disabled here. |
| MLFLOW_EXPERIMENT_NAME           | fraud_detection_demo                                          | The unique MLflow experiment name of your project.  This is normally the same as the project name                 |
| ADSP_WORKER_MAX                  | 1                                                             | The per-project maximum number of background jobs a user can leverage during parallel execution                   |
| SELF_HOSTED_MODEL_ENDPOINT       | https://demo-fraud-detection-api.anaconda.example.com         | Prediction API Endpoint URL                                                                                       |
| BOKEH_ALLOW_WS_ORIGIN            | demo-fraud-detection-dashboard.anaconda.example.com           | F.Q.D.N. for the Dashboard URL                                                                                    |
| SELF_HOSTED_DATA_STREAM          | https://demo-fraud-detection-data-stream.anaconda.example.com | Data Stream API Endpoint URL                                                                                      |

### Project Configuration Secrets / Variables
These are only required if utilizing private / shared deployments.
* These **SHOULD** be created as AE5 User Secrets rather than as project environment variables.
* For additional details see [Storing Secrets](https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/user-settings.html?highlight=secrets#storing-secrets).

| Name                             | Description                              |
|----------------------------------|------------------------------------------|
| SELF_HOSTED_MODEL_ENDPOINT_TOKEN | Private access token for Prediction API  |
| SELF_HOSTED_DATA_STREAM_TOKEN    | Private access token for Data Stream API |

### File System Data Configuration
| Name                             | Default                                                       | Description                                                                      |
|----------------------------------|:--------------------------------------------------------------|----------------------------------------------------------------------------------|
| DATA_BASE_DIR                    | data                                                          | The folder to look for data under.  In ADSP this would normally always be `data` |
| DATA_ARTIFACT                    | creditcard.csv                                                | The name of CSV file containing our source data                                  |
| TRUTH_COLUMN_NAME                | Class                                                         | The column name for the truth labels                                             |


### Model Training Configuration

| Name                             | Default                                                       | Description                                                             |
|----------------------------------|:--------------------------------------------------------------|-------------------------------------------------------------------------|
| SELF_HOSTED_MODEL_AUTO_PROMOTION | "True"                                                        | Allow the workflow to move the alias to a new model with a better score |
| SELF_HOSTED_MODEL_ALIAS          | production                                                    | Alias to manage during champion /challenger promotion                   |
