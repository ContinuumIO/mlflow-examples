""" Dashboard State [Model] """

from __future__ import annotations

import pandas as pd

from fraud_detection.client.dashboard import DashboardClient
from fraud_detection.client.fraud_detection import FraudDetectionClient
from fraud_detection.client.stream import StreamClient
from fraud_detection.contracts.dto.dashboard.client_response import DashboardClientResponse
from fraud_detection.contracts.dto.dashboard.mlflow_reports import DashboardMLflowReport
from fraud_detection.report.mlflow_reporting import MLflowReporting
from fraud_detection.services.mlflow_helper import mlflow_environment_config


class DashboardState:
    """This class holds the Dashboard state (model) that the view renders."""

    def __init__(self):
        mlflow_reporting: MLflowReporting = MLflowReporting(client=mlflow_environment_config().client)

        stream_client: StreamClient = StreamClient(auth=False, secure=False)
        fraud_detection_client: FraudDetectionClient = FraudDetectionClient()

        self.client: DashboardClient = DashboardClient(
            stream_client=stream_client, fraud_detection_client=fraud_detection_client, mlflow_reporting=mlflow_reporting
        )

        self.transactions: pd.DataFrame = pd.DataFrame([])
        self.features: pd.DataFrame = pd.DataFrame([])
        self.fraud_review_cases: pd.DataFrame = pd.DataFrame([])
        self.mlflow_workflow_runs: pd.DataFrame = pd.DataFrame([])
        self.mlflow_models: pd.DataFrame = pd.DataFrame([])

    def reset(self):
        """Resets the Stream API and Dashboard state."""

        self.transactions = pd.DataFrame([])
        self.features = pd.DataFrame([])
        self.fraud_review_cases = pd.DataFrame([])
        self.mlflow_workflow_runs: pd.DataFrame = pd.DataFrame([])
        self.mlflow_models: pd.DataFrame = pd.DataFrame([])

        self.client.stream_client.reset()

    def update_mlflow_data(self, max_rows: int = 50) -> None:
        """
        Updates the MLflow report state.

        Parameters
        ----------
        max_rows: int = 50
            The maximum number of rows in the reports
        """

        reports: DashboardMLflowReport = self.client.get_mlflow_reports()
        self.mlflow_workflow_runs = reports.workflow[:max_rows]
        self.mlflow_models = reports.model[:max_rows]

    def update(self, max_rows: int = 500):
        """
        Updates the stream data state.

        Parameters
        ----------
        max_rows: int = 500
            The maximum number of rows in the reports
        """

        # Get response
        response: DashboardClientResponse = self.client.get_data()

        # preprocess
        features_row: list | None = response.features

        # post-process transactions
        self.transactions = DashboardState._process_df(df=self.transactions, row=[response.dict(exclude_none=True, exclude={"features", "label"})])

        # Add cases to review if they are suspect.
        if response.prediction != 0:
            self.fraud_review_cases = DashboardState._process_df(
                df=self.fraud_review_cases, row=[response.dict(exclude_none=True, exclude={"features", "label"})], max_rows=max_rows
            )

        # post-process features
        if features_row:
            self.features = DashboardState._process_df(df=self.features, row=features_row, max_rows=max_rows)

    # pylint: disable=invalid-name
    @staticmethod
    def _process_df(df: pd.DataFrame, row: list, max_rows: int = 500) -> pd.DataFrame:
        """
        Adds & returns the new row to the top (0 index) of the dataframe and truncates the result as needed.

        Parameters
        ----------
        df: pd.DataFrame
            Source dataframe
        row: list
            The new row.
        max_rows: int = 500
            The maximum number of rows in the reports

        Returns
        -------
        updated_dataframe: pd.DataFrame
        """

        row_df = pd.DataFrame.from_dict(row)
        new_df = pd.concat([row_df, df], ignore_index=True)
        new_df = new_df[:max_rows]
        return new_df
