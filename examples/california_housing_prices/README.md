# California Housing Prices

## About Dataset

* Source: https://www.kaggle.com/datasets/camnugent/california-housing-prices
* License: https://creativecommons.org/publicdomain/zero/1.0/

### Content
The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. Be warned the data aren't cleaned so there are some preprocessing steps required! The columns are as follows, their names are pretty self explanitory:

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* median_house_value
* ocean_proximity

## Setup
1. Create an AE5 user secret `MLFLOW_TRACKING_TOKEN` and populate it with the latest private access token for the MLflow deployment.
2. Update `anaconda-project.yml`:
   1. Update `SELF_HOSTED_MODEL_ENDPOINT` to reference the expected end-point name. 
   2. Update `BOKEH_ALLOW_WS_ORIGIN` to point to the expected Dashboard deployment URL.
3. Upload the project to AE5:
    > cd examples/california_housing_prices

    > ae5 project upload . 

## How to use this demo?
1. Train the models using the `train-{elasticnet,xgboost}` notebooks.
2. Review model performance with `model-comparision` notebook.
3. Deploy a REST API with the `Production` model using an AE5 Deployment.
4. Deploy the dashboard (which consumes the API).
