"""
This module contains data related helper functions.
"""

import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split


class DataSet(BaseModel):
    """DataSet DTO"""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    class Config:
        """Pydantic class config override"""

        arbitrary_types_allowed = True


def prepare_data(csv_url: str) -> DataSet:
    """
    Loads the data from csv file, and returns train, test splits for training.

    Parameters
    ----------
    csv_url: str
        The location of the CSV file to load.

    Returns
    -------
    ds: DataSet
        An instance of a DataSet DTO.
    """

    (X, y) = load_data(csv_url=csv_url, truth_col_name="quality")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return DataSet(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def load_data(csv_url: str, truth_col_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads features and truth data from specified CSV file and truth column.

    Parameters
    ----------
    csv_url: str
        The location of the CSV file to load.
    truth_col_name: str
        The name of column which contains the truth data.

    Returns
    -------
    tuple
        A tuple of (X, y)
    """

    data: pd.DataFrame = pd.read_csv(csv_url, sep=",")
    X: pd.DataFrame = data.drop([truth_col_name], axis=1)
    y: pd.DataFrame = data[[truth_col_name]]
    return X, y
