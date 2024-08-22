# Commands

### Development

These commands are used during project development.

| Command          | Environment | Description                                         |
|------------------|-------------|:----------------------------------------------------|
| bash             | Default     | Run bash within the `default` environment           |
| clean            | Default     | Cleanup temporary project files                     |
| lint             | Default     | Perform code linting check                          |
| lint:fix         | Default     | Perform automated code formatting                   |
| start:jupyterlab | Default     | Starts Jupyter Lab within the `default` environment |

### Setup

These commands are used during the initial setup of the project.

| Command   | Environment | Description                                |
|-----------|-------------|:-------------------------------------------|
| bootstrap | Default     | Environment preparation for model training |


### Runtime

These commands are used to execute the different components of the project.

| Command             | Environment | Description                                                                                                   |
|---------------------|-------------|:--------------------------------------------------------------------------------------------------------------|
| dashboard           | Default     | Dashboard - Consumes the Prediction API, Data Stream API, and the MLflow API to provide a client consumer.    |
| prediction-api      | Default     | Prediction API - Deployments created from this command will serve the `production` model REST API.            |
| data-stream         | Default     | Emits time series data suitable for inference.                                                                |
| workflow:main:adsp  | Default     | ADSP Integrated Workflow [Training] Execution - This command is intended to be leveraged by a scheduled job.  |
| workflow:main:local | Default     | Local Workflow [Training] Execution - This command is intended to be run locally during workflow development. |

### Notebooks

These commands are used during development for solution management.

| Command                    | Environment | Description                                                                                              |
|----------------------------|-------------|:---------------------------------------------------------------------------------------------------------|
| exploration                | Default     | Initial data exploration, and statistics                                                                 |
| train-with-under-sampling  | Default     | ML models (Decision Trees, and Random Forests) Training pipeline with Near-Miss under-sampling technique |
| train-with-over-sampling   | Default     | ML models (Decision Trees, and Random Forests) Training pipeline with SMOTE over-sampling technique      |
| evaluate-time-performance  | Default     | Evaluate Timing Execution Performance of ML Pipelines                                                    |
| feature-importance         | Default     | Visualize Decision Tree and Random Forest feature importance                                             |
| evaluate-model-performance | Default     | Evaluate ML Model Performance (F1 Score)                                                                 |

### Background Worker

These commands are used by the ADSP plugin when executing workflow steps as background jobs. Lease as-is.

| Command | Environment      | Description                                                                         |
|---------|------------------|:------------------------------------------------------------------------------------|
| Worker  | worker_bootstrap | Used by the ADSP MLflow Plugin for executing workflow steps with the job scheduler. |
