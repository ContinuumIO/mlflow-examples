import os


def upsert_environment() -> str:
    experiment_name: Optional[str] = None
    if "MLFLOW_EXPERIMENT_NAME" in os.environ:
        experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]

    try:
        experiment = mlflow.get_experiment_by_name(name=experiment_name)
        experiment_id: str = experiment.experiment_id
    except Exception as error:
        print(error)
        print("Creating experiment")
        experiment_id: str = mlflow.create_experiment(name=experiment_name)

    return experiment_id
