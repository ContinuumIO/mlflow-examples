"""Module holds MLflow reports"""

import datetime

import mlflow
import pandas as pd
from ae5_tools import demand_env_var
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from sklearn.metrics import classification_report

from fraud_detection.client.data import DataClient
from fraud_detection.client.mlflow_client import AnacondaMlFlowClient


class MLflowReporting(AnacondaMlFlowClient):
    """MLflow reporting generation class."""

    def build_workflow_runs_report(self) -> pd.DataFrame:
        """
        Builds the report for training workflow runs.

        Returns
        -------
        workflow_report: pd.DataFrame
            Workflow Report
        """

        # Get experiment runs
        runs: list[Run] = self.get_experiment_runs(
            experiment_id=self.experiment.experiment_id,
            filter_string="attributes.run_name LIKE 'workflow-fraud-detection-model-training-parallel-%'",
            order_by=["start_time DESC", "end_time DESC", "run_id"],
        )

        workflow_report: pd.DataFrame = pd.DataFrame([])
        for run in runs:
            row: dict = {
                "name": [run.info.run_name],
                "start": [datetime.datetime.fromtimestamp(run.info.start_time / 1000)],
                "end": [datetime.datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None],
                "status": [run.info.status],
                "run_id": [run.info.run_id],
            }

            row_df: pd.DataFrame = pd.DataFrame.from_dict(row)
            workflow_report = pd.concat([row_df, workflow_report], ignore_index=True)
        if "start" in workflow_report.columns:
            workflow_report.sort_values(by=["start"], inplace=True, ascending=False)
        return workflow_report

    def build_model_registry_report(self) -> pd.DataFrame:
        """
        Builds the report for models.

        Returns
        -------
        model_report: pd.DataFrame
            Model Report
        """

        model_report: pd.DataFrame = pd.DataFrame([])
        versions: list[ModelVersion] = self.get_model_versions(filter_string=f"name = '{self.experiment.name}'")

        aliased_versions: dict = {}
        aliases: list[str] = [demand_env_var(name="SELF_HOSTED_MODEL_ALIAS")]
        for alias in aliases:
            aliased_model_version: ModelVersion = self.client.get_model_version_by_alias(name=self.experiment.name, alias=alias)
            aliased_versions[alias] = aliased_model_version.version

        for version in versions:
            row: dict = {
                "name": [version.name],
                "version": [version.version],
                "aliases": [""],
                "score": [""],
                "run_id": [version.run_id],
                "tags": [version.tags],
                "created": [datetime.datetime.fromtimestamp(version.creation_timestamp / 1000)],
            }

            # post process aliases
            row_aliases: list[str] = [key for key, value in aliased_versions.items() if value == version.version]
            if len(row_aliases) > 0:
                row["aliases"] = row_aliases

            # post process tags
            if "best_cv_score_f1" in version.tags:
                row["score"] = version.tags["best_cv_score_f1"]

            row_df: pd.DataFrame = pd.DataFrame.from_dict(row)
            model_report = pd.concat([row_df, model_report], ignore_index=True)
            model_report.sort_values(by=["score"], inplace=True, ascending=False)

        return model_report

    def _get_challenger_version(self, run_id: str) -> ModelVersion:
        """
        Get the Challenger Model Version for the specified run.

        Parameters
        ----------
        run_id: str
            Ru Id to pull the challenger from.

        Returns
        -------
        challenger: ModelVersion
            The top model (by score) from the set of models trained.
        """

        # Get experiment runs
        runs: pd.DataFrame = mlflow.search_runs(experiment_ids=[self.experiment.experiment_id], filter_string=f"tags.mlflow.parentRunId='{run_id}'")

        # Get the four child runs (model x sampling) run results
        filtered_runs: pd.DataFrame = runs[(runs["status"] == "FINISHED") & (runs["metrics.best_cv_score"].notnull())]

        # Get top run
        challenger_run: pd.DataFrame = filtered_runs.sort_values(by="metrics.best_cv_score", ascending=False)[0:1]
        challenger_run_id: str = challenger_run["run_id"].iloc[0]
        return mlflow.search_model_versions(filter_string=f"tags.run_id='{challenger_run_id}'")[0]

    @staticmethod
    def _get_model_accuracies(data_client: DataClient, challenger_model, champion_model) -> dict:
        """
        Generate accuracy scores for the models.

        Parameters
        ----------
        data_client: DataClient
            Data client for input
        challenger_model
            Challanger model
        champion_model
            Champion model

        Returns
        -------
        accuracy_report: dict
            Model accuracy report
        """

        # pylint: disable=invalid-name
        # Load data
        X, y_truth, labels = data_client.load_data()

        # Get Challenger F1 score
        challenger_y_predicted: pd.DataFrame = pd.DataFrame(challenger_model.predict(X), columns=[data_client.truth_col_name])
        challenger_perf: dict = classification_report(y_true=y_truth, y_pred=challenger_y_predicted, labels=labels, output_dict=True)
        challenger_accuracy: float = challenger_perf["accuracy"]

        # Get Current Champion F1 score
        champion_y_predicted: pd.DataFrame = pd.DataFrame(champion_model.predict(X), columns=[data_client.truth_col_name])
        champion_perf: dict = classification_report(y_true=y_truth, y_pred=champion_y_predicted, labels=labels, output_dict=True)
        champion_accuracy: float = champion_perf["accuracy"]

        return {"accuracy": {"challenger": challenger_accuracy, "champion": champion_accuracy}}

    def champion_challenger_report(self, run_id: str, data_client: DataClient) -> str:
        """
        Performs Champion / Challenger model performance comparison. Returns the registered model version of the winner.

        Parameters
        ----------
        run_id: str
            The parent run id to perform analysis on.
        data_client: DataClient
            data client for data

        Returns
        -------
        version: str
            The model version of the Champion.
        """

        # Get challenger version
        challenger_model_version: ModelVersion = self._get_challenger_version(run_id=run_id)

        # Ensure there is an alias assigned version (otherwise this becomes the first)
        try:
            champion_model_version: ModelVersion = self.client.get_model_version_by_alias(
                name=demand_env_var(name="MLFLOW_EXPERIMENT_NAME"), alias=demand_env_var(name="SELF_HOSTED_MODEL_ALIAS")
            )
        except mlflow.exceptions.RestException as error:
            if error.error_code == "INVALID_PARAMETER_VALUE":
                # no current version is set, local champion becomes global
                return challenger_model_version.version
            raise error from error

        # Build registered model [version] URI for the challenger run
        challenger_model_uri: str = f"models:/{demand_env_var(name='MLFLOW_EXPERIMENT_NAME')}/{challenger_model_version.version}"

        # Load challenger model
        challenger_model = mlflow.pyfunc.load_model(challenger_model_uri, suppress_warnings=True)

        # Build the champion model [version] URI
        champion_model_uri: str = f"models:/{demand_env_var(name='MLFLOW_EXPERIMENT_NAME')}/{champion_model_version.version}"

        # Load the champion model
        champion_model = mlflow.pyfunc.load_model(champion_model_uri, suppress_warnings=True)

        # Compete
        competition_results: dict = MLflowReporting._get_model_accuracies(
            data_client=data_client, challenger_model=challenger_model, champion_model=champion_model
        )
        challenger_accuracy: float = competition_results["accuracy"]["challenger"]
        champion_accuracy: float = competition_results["accuracy"]["champion"]

        if challenger_accuracy > champion_accuracy:
            # promote the champion
            return challenger_model_version.version

        # Keep the current champion
        return champion_model_version.version
