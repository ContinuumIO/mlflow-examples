""" This module contains REST client for the fraud detection endpoint. """

import pandas as pd
from ae5_tools import demand_env_var
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from fraud_detection.contracts.dto.abstract import BaseModel
from fraud_detection.contracts.errors.request_failure_error import RequestFailureError


class FraudDetectionClient(BaseModel):
    """Client interface for our Fraud Detection REST API."""

    @staticmethod
    def _invoke_rest_endpoint(input_data: dict, auth: bool = True) -> dict:
        """
        Invokes the REST endpoint.

        Parameters
        ----------
        input_data: dict
            The data to POST to the endpoint.
        auth: bool
            Flag for providing bearer token.

        Returns
        -------
        response: dict
            The response from the API, raises Exception under failure conditions.
        """

        headers: dict = {}
        endpoint_url: str = demand_env_var(name="SELF_HOSTED_MODEL_ENDPOINT")
        if auth:
            headers: dict = {"Authorization": f"Bearer {demand_env_var(name='SELF_HOSTED_MODEL_ENDPOINT_TOKEN')}"}

        session: Session = Session()
        retries: Retry = Retry(
            total=10,
            backoff_factor=0.1,
            status_forcelist=[502, 503, 504],
            allowed_methods={"POST"},
        )
        adapter: HTTPAdapter = HTTPAdapter(max_retries=retries)
        session.mount(prefix="https://", adapter=adapter)

        post_params: dict = {
            "url": f"{endpoint_url}/invocations",
            "json": input_data,
            "verify": False,
            "headers": headers,
            "timeout": 30,
        }

        response = session.post(**post_params)
        if response.status_code != 200:
            message: str = f"Received status code: ({response.status_code}), Failed to call prediction: {response.text}"
            raise RequestFailureError(message=message, status_code=response.status_code, request=post_params)
        return response.json()

    @staticmethod
    def predict(data_x: pd.DataFrame, auth: bool = True) -> pd.DataFrame:
        """
        Get prediction for the given input.

        Parameters
        ----------
        data_x: pd.DataFrame
            The feature data to predict on.
        auth: bool
            Default: True
            Control flag for the inclusion of a Bearer token.

        Returns
        -------
        y_pred: pd.DataFrame
            A dataframe of predictions.
        """

        # Build the prediction request
        params: dict = {
            "input_data": {"dataframe_records": data_x.to_dict(orient="records")},
            "auth": auth,
        }

        # Call prediction service
        y_pred_dict: dict = FraudDetectionClient._invoke_rest_endpoint(**params)
        return pd.DataFrame(y_pred_dict)
