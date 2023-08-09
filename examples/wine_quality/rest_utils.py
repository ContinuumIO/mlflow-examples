"""
This module contains REST helper functions.
"""

from typing import Optional

import pandas as pd
import requests

from ae5_tools import demand_env_var


def invoke_rest_endpoint(endpoint_url: str, input_data: dict) -> dict:
    """
    Invokes the REST endpoint.

    Parameters
    ----------
    endpoint_url: str
        The URL of the REST endpoint.
    input_data: dict
        The data to POST to the endpoint.

    Returns
    -------
    response: dict
        The response from the API, raises Exception under failure conditions.
    """

    response = requests.post(url=f"{endpoint_url}/invocations", json=input_data, verify=False)
    if response.status_code != 200:
        raise Exception(f"Received status code: ({response.status_code}), Failed to call prediction: {response.text}")
    return response.json()


def predict(endpoint_url: Optional[str], data_x: pd.DataFrame) -> pd.DataFrame:
    """
    Get prediction for the given input.

    Parameters
    ----------
    endpoint_url: str
        The URL of the REST endpoint.
    data_x: pd.DataFrame
        The feature data to predict on.

    Returns
    -------
    y_pred: pd.DataFrame
        A dataframe of predictions.
    """

    endpoint_url = endpoint_url if endpoint_url else demand_env_var(name="SELF_HOSTED_MODEL_ENDPOINT")

    # Build the prediction request
    params: dict = {
        "endpoint_url": endpoint_url,
        "input_data": {"dataframe_records": data_x.to_dict(orient="records")},
    }

    # Call prediction service
    y_pred_dict: dict = invoke_rest_endpoint(**params)
    return pd.DataFrame(y_pred_dict)
