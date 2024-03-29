# xgboost sklearn-learn and MLflow Demo


## Setup
1. Create an AE5 user secret `MLFLOW_TRACKING_TOKEN` and populate it with the latest private access token for the MLflow deployment.
2. Update `anaconda-project.yml`:
   1. Update `SELF_HOSTED_MODEL_ENDPOINT` to reference the expected end-point name. 
3. Upload the project to AE5:
    > cd examples/wine_quality

    > ae5 project upload . 

## How to use this demo?
1. Train the models using the `train-{elasticnet,xgboost}` notebook.
2. Review model performance with `model-comparision` notebook.
2. Deploy a REST API with the `Production` model.
3. Deploy the wine quality dashboard.
