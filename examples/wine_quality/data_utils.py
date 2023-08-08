import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split


class DataSet(BaseModel):
    """
    Data Preparation
    Loads the data from csv file, and returns our train, test splits for training.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


def prepare_data(csv_url: str) -> DataSet:
    data: pd.DataFrame = pd.read_csv(csv_url, sep=",")

    # The predicted column is `quality`, which is a scalar from [3, 9]
    X: pd.DataFrame = data.drop(["quality"], axis=1)
    y: pd.DataFrame = data[["quality"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return DataSet(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
