import pandas as pd
import requests

from ae5_tools import demand_env_var


def invoke_rest_endpoint(endpoint_url: str, input_data: dict) -> dict:
    """
    Invokes the REST Endpoint
    """

    response = requests.post(url=f"{endpoint_url}/invocations", json=input_data, verify=False)
    if response.status_code != 200:
        raise Exception(f"Received status code: ({response.status_code}), Failed to call prediction: {response.text}")
    return response.json()


def predict(data_x: pd.DataFrame) -> pd.DataFrame:
    """
    Get prediction for the  given input.

    return y_pred
        predictions
    """

    # Build the prediction request
    params: dict = {
        "endpoint_url": demand_env_var(name="SELF_HOSTED_MODEL_ENDPOINT"),
        "input_data": {"dataframe_records": data_x.to_dict(orient="records")},
    }

    # Call prediction service
    y_pred_dict: dict = invoke_rest_endpoint(**params)
    return pd.DataFrame(y_pred_dict)
