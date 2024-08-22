"""Dashboard Client Definition"""

from __future__ import annotations

import pandas as pd
from ae5_tools import demand_env_var

from fraud_detection.client.fraud_detection import FraudDetectionClient
from fraud_detection.client.stream import StreamClient
from fraud_detection.contracts.dto.abstract import BaseModel
from fraud_detection.contracts.dto.dashboard.client_response import DashboardClientResponse
from fraud_detection.contracts.dto.dashboard.mlflow_reports import DashboardMLflowReport
from fraud_detection.contracts.dto.stream_data import StreamData
from fraud_detection.report.mlflow_reporting import MLflowReporting


# pylint: disable=invalid-name
class DashboardClient(BaseModel):
    """
    Dashboard Client
    Used by the Dashboard Panel to access data.

    stream_client: StreamClient
        Stream API Client
    fraud_detection_client: FraudDetectionClient
        Fraud Detection API Client
    mlflow_reporting: MLflowReporting
        MLflow Reporting
    """

    stream_client: StreamClient
    fraud_detection_client: FraudDetectionClient
    mlflow_reporting: MLflowReporting

    def get_data(self) -> DashboardClientResponse:
        """
        Returns a data sample and truth label (if available) from the data source.

        Returns
        -------
        response: DashboardClientResponse
        """

        stream_data: StreamData = self._get_X()
        prediction = self.fraud_detection_client.predict(data_x=stream_data.X, auth=False)
        predicted_label: int = prediction["predictions"].iloc[0]
        return DashboardClient._build_response(X=stream_data.X, true_label=stream_data.true_label, predicted_label=predicted_label)

    def get_mlflow_reports(self) -> DashboardMLflowReport:
        """
        Returns the MLflow reports for the project.

        Returns
        -------
        report: DashboardMLflowReport
        """

        return DashboardMLflowReport(
            workflow=self.mlflow_reporting.build_workflow_runs_report(), model=self.mlflow_reporting.build_model_registry_report()
        )

    def _get_X(self) -> StreamData:
        """
        Gets `X` value from the stream.

        Returns
        -------
        data: StreamData
            An instance of `StreamData` for the received data.
        """

        X = self.stream_client.sample()

        true_label: int | None = None
        truth_column_name: str = demand_env_var(name="TRUTH_COLUMN_NAME")
        if truth_column_name in X:
            true_label: int = X[truth_column_name]
        del X[truth_column_name]

        X_df = pd.DataFrame.from_dict(data=[X])
        return StreamData(X=X_df, true_label=true_label)

    @staticmethod
    def _build_response(X: pd.DataFrame, true_label: int | None, predicted_label: int) -> DashboardClientResponse:
        """
        Builds and returns the DashboardClientResponse.

        Parameters
        ----------
        X: pd.DataFrame
            A single sample which can be marshalled into an input Tensor for the model.
        true_label: int | None
            The true label of the data sample (if available).
        predicted_label: int
            The predicted label.

        Returns
        -------
        response: DashboardClientResponse
        """

        response: dict = {"time": X["Time"].iloc[0], "amount": X["Amount"].iloc[0], "label": true_label, "prediction": predicted_label}
        features_df = X.drop(["Time", "Amount"], axis=1)
        response["features"] = [features_df.to_dict(orient="list")]
        return DashboardClientResponse.model_validate(response)
