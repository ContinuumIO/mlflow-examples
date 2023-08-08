import mlflow
import mlflow.sklearn
from mlflow.entities import Experiment
from mlflow.tracking import MlflowClient
from mlflow_utils import upsert_model_registry

from ae5_tools import load_ae5_user_secrets
from mlflow_adsp import upsert_experiment


def init() -> tuple[str, MlflowClient]:
    # Load user specific configuration.
    load_ae5_user_secrets()

    # Generate a client, this will be used for several operations across the notebook.
    client = MlflowClient()

    experiment: Experiment = mlflow.set_experiment(experiment_id=upsert_experiment())
    upsert_model_registry(client=client)

    return experiment.experiment_id, client
