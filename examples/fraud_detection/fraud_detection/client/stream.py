""" Streaming Data Service Client Definition """

from __future__ import annotations

from ae5_tools import demand_env_var
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from fraud_detection.contracts.errors.request_failure_error import RequestFailureError
from fraud_detection.contracts.types.request_verb import RequestVerb


class StreamClient:
    """Data Stream Client"""

    def __init__(self, auth: bool = True, secure: bool = True):
        """
        Parameters
        ----------
        auth: bool = True
            Flag for including bearer token.
        secure: bool = True
            Flag for using TLS.
        """

        self.auth = auth
        self.secure: bool = secure

        self.session: Session = Session()

        self.headers: dict = {}
        self.endpoint_url: str = demand_env_var(name="SELF_HOSTED_DATA_STREAM")
        if self.auth:
            self.headers: dict = {"Authorization": f"Bearer {demand_env_var(name='SELF_HOSTED_DATA_STREAM_TOKEN')}"}

        retries: Retry = Retry(
            total=10,
            backoff_factor=0.1,
            status_forcelist=[502, 503, 504],
            allowed_methods={"GET", "PUT"},
        )

        adapter: HTTPAdapter = HTTPAdapter(max_retries=retries)

        if self.secure:
            self.session.mount(prefix="https://", adapter=adapter)
        else:
            self.session.mount(prefix="http://", adapter=adapter)

    def _invoke_rest_endpoint(self, path: str, verb: RequestVerb, input_data: dict | None = None, params: dict | None = None) -> dict | list:
        """
        Invokes the REST endpoint.

        Parameters
        ----------
        path: str
            The path.
        verb: RequestVerb
            The REST verb.
        input_data: dict | None = None
            The data to POST to the endpoint.
        params: dict | None = None
            The parameters for the call.

        Returns
        -------
        response: dict | list
            The response from the API, raises Exception under failure conditions.
        """

        post_params: dict = {
            "url": f"{self.endpoint_url}/{path}",
            "verify": False,
            "headers": self.headers,
            "timeout": 30,
        }

        if input_data:
            post_params["json"] = input_data

        if params:
            post_params["params"] = params

        if verb == RequestVerb.GET:
            response = self.session.get(**post_params)
        elif verb == RequestVerb.PUT:
            response = self.session.put(**post_params)
        elif verb == RequestVerb.POST:
            response = self.session.post(**post_params)
        elif verb == RequestVerb.DELETE:
            response = self.session.delete(**post_params)
        else:
            message: str = f"verb {verb} not yet implemented"
            raise NotImplementedError(message)

        if int(response.status_code) not in [200, 201]:
            message: str = f"Received status code: ({response.status_code}), Response: {response.text}"
            raise RequestFailureError(message=message, status_code=response.status_code, request=post_params)
        return response.json()

    def sample(self) -> dict:
        """
        Gets the next input from the time series data.

        Returns
        -------
        sample: dict
            Sample data.
        """

        return self._invoke_rest_endpoint(path="api/v1/sample", verb=RequestVerb.GET)

    def reset(self) -> None:
        """Resets the stream API time index."""

        self._invoke_rest_endpoint(path="api/v1/reset", verb=RequestVerb.PUT)

    def index(self, value: int) -> dict:
        """
        Returns the time series data at the specific index.

        Parameters
        ----------
        value: int
            The value of the index

        Returns
        -------
        sample: dict
            Sample data from the specific index.
        """

        return self._invoke_rest_endpoint(path="api/v1/index", params={"value": value}, verb=RequestVerb.GET)

    def batch(self, start: int, size: int) -> list:
        """
        Returns a batch of data from the stream.

        Parameters
        ----------
        start: int
            The index to start the batch from.
        size: int
            The size of the batch to request.

        Returns
        -------
        batch: list
            A list of sample data as dictionaries.
        """

        return list(self._invoke_rest_endpoint(path="api/v1/batch", params={"start": start, "size": size}, verb=RequestVerb.GET))

    def info(self) -> dict:
        """
        Returns state information from the Data Stream API.

        Returns
        -------
        info: dict
            dictionary of status details.
        """

        return self._invoke_rest_endpoint(path="api/v1/info", verb=RequestVerb.GET)
