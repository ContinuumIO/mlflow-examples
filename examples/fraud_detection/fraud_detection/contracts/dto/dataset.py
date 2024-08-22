"""This module contains definitions for our Data Set structure."""

import pandas as pd

from fraud_detection.contracts.dto.abstract import BaseModel


class DataSet(BaseModel):
    """
    DataSet DTO

    Attributes
    ----------
    X_train: pd.DataFrame
        Training Data
    y_train: pd.Series
        Training Data Labels
    X_test: pd.DataFrame
        Test Data
    y_test: pd.Series
        Test Data Labels
    labels: pd.Series
         Complete set of data labels.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    labels: pd.Series
