""" This module provides an interface for MLFlow Tracking Operations. """

from __future__ import annotations

import os
from typing import Any

import mlflow
from ae5_tools import demand_env_var
from mlflow import MlflowClient, MlflowException
from mlflow.entities import Experiment, Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from mlflow.store.entities import PagedList
from mlflow_adsp import upsert_experiment

from fraud_detection.contracts.dto.abstract import BaseModel


class AnacondaMlFlowClient(BaseModel):
    """
    This class provides an interface for interacting with the MLFlow Tracking Server.

    Attributes
    ----------
    client: MlflowClient
        An instance of a raw mlflow client.
    """

    client: MlflowClient
    experiment: Experiment | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.experiment: Experiment = mlflow.set_experiment(experiment_id=upsert_experiment())

    @staticmethod
    def _get_context_experiment() -> Experiment:
        # Get our experiment
        experiment_name: str = demand_env_var(name="MLFLOW_EXPERIMENT_NAME")
        return mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")[0]

    def upsert_model_registry(self) -> None:
        """
        Upsert (Update or Insert) a model registry into MLflow.
        """

        try:
            self.client.create_registered_model(name=os.environ["MLFLOW_EXPERIMENT_NAME"])
        except (MlflowException, RestException) as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise error

    def get_experiment_runs(self, experiment_id: str, filter_string: str | None = None, order_by: list[str] | None = None) -> list[Run]:
        """
        Consumes the paged MLFlow runs API and returns a consolidated list of found runs.

        Parameters
        ----------
        experiment_id: str
            The experiment id to limit search within.
        filter_string: str | None = None
            An optional filter string for the query.
        order_by: list[str] | None = None
            An optional order by clause.

        Returns
        -------
        experiments: list[Run]
            A list of `Run` objects found from the search.
        """

        results: PagedList[Run] = PagedList(items=[], token=None)

        halt_paging: bool = False
        page_token: str | None = None
        while not halt_paging:
            reported_runs: PagedList[Run] = self.client.search_runs(
                experiment_ids=[experiment_id], page_token=page_token, filter_string=filter_string, order_by=order_by
            )
            if reported_runs.token is not None:
                page_token = reported_runs.token
            else:
                halt_paging = True
            results += reported_runs

        return list(results)

    def get_model_versions(self, filter_string: str | None = None) -> list[ModelVersion]:
        """
        Returns all model versions for the specified model name.

        Parameters
        ----------
        filter_string: str | None = None
            An optional filter string for the query.

        Returns
        -------
        versions: PagedList[ModelVersion]
            A paged list of all model versions for the specified model.
        """

        model_versions: PagedList[ModelVersion] = PagedList(items=[], token=None)
        halt_paging: bool = False
        page_token: str | None = None
        while not halt_paging:
            model_versions_paged: PagedList[ModelVersion] = self.client.search_model_versions(page_token=page_token, filter_string=filter_string)
            if model_versions_paged.token is not None and model_versions_paged.token != "":
                page_token = model_versions_paged.token
            else:
                halt_paging = True
            model_versions += model_versions_paged

        return list(model_versions)

    def register_model(self, run: Run, tags: dict | None = None, suffix: str = "model") -> ModelVersion:
        """
        Registers the model tracked on the provided run.

        Parameters
        ----------
        run: Run
            Instance of an MLflow Run with a model logged to it.
        tags: dict | None = None
            A dictionary of additional tags.
        suffix: str
            Default: "model"
            The `folder` name to store the model under within artifact storage. Leave as `model` under most conditions.

        Returns
        -------
        version: ModelVersion
            The model version created during registration.
        """

        # Ensure we are ready to track models
        self.upsert_model_registry()

        final_tags = {"run_id": run.info.run_id}
        if tags:
            final_tags = {**tags, **final_tags}

        model_version: ModelVersion = self.client.create_model_version(
            name=os.environ["MLFLOW_EXPERIMENT_NAME"],
            source=f"{run.info.artifact_uri}/{suffix}",
            run_id=run.info.run_id,
            tags=final_tags,
        )
        return model_version
