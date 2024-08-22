"""This module contains data related helper functions."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.common.constants import rng
from fraud_detection.contracts.dto.abstract import BaseModel
from fraud_detection.contracts.dto.dataset import DataSet


class DataClient(BaseModel):
    """Client Data Interface"""

    csv_url: Path
    truth_col_name: str

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Loads features and truth data from specified CSV file and truth column.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            tuple of (X, y, labels)
        """

        # pylint: disable=invalid-name

        # load data set
        df = pd.read_csv(self.csv_url)
        X, y = df[df.columns[df.columns != self.truth_col_name]], df[self.truth_col_name]
        labels: pd.Series = y.drop_duplicates()
        return X, y, labels

    def get_dataset(self) -> DataSet:
        """
        Loads features and truth data from specified CSV file and truth column.

        Returns
        -------
        dataset: DataSet
            DataSet DTO of data
        """

        # pylint: disable=invalid-name

        # load data set
        X, y, labels = self.load_data()

        # split and marshall into DTO
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=rng)
        return DataSet(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, labels=labels)
