"""
This module contains data related helper functions.
"""
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


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

    (X, y) = load_data(csv_url=csv_url, truth_col_name="median_house_value")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return DataSet(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# function that imputes a dataframe
def impute_knn(df: pd.DataFrame) -> pd.DataFrame:
    # From https://www.kaggle.com/code/shtrausslearning/bayesian-regression-house-price-prediction
    # license: LICENSE.apache
    """inputs: pandas df containing feature matrix"""
    """ outputs: dataframe with NaN imputed """
    # imputation with KNN unsupervised method

    # separate dataframe into numerical/categorical
    ldf = df.select_dtypes(include=[np.number])  # select numerical columns in df
    ldf_putaside = df.select_dtypes(exclude=[np.number])  # select categorical columns in df
    # define columns w/ and w/o missing data
    cols_nan = ldf.columns[ldf.isna().any()].tolist()  # columns w/ nan
    cols_no_nan = ldf.columns.difference(cols_nan).values  # columns w/o nan

    for col in cols_nan:
        imp_test = ldf[ldf[col].isna()]  # indicies which have missing data will become our test set
        imp_train = ldf.dropna()  # all indicies which which have no missing data
        model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
        knr = model.fit(imp_train[cols_no_nan], imp_train[col])
        ldf.loc[df[col].isna(), col] = knr.predict(imp_test[cols_no_nan])

    return pd.concat([ldf, ldf_putaside], axis=1)


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    # From https://www.kaggle.com/code/shtrausslearning/bayesian-regression-house-price-prediction
    # license: LICENSE.apache

    # Address missing data
    data = impute_knn(df=data)

    # Remove outliners
    maxval2: pd.DataFrame = data["median_house_value"].max()  # get the maximum value
    data = data[data["median_house_value"] != maxval2]

    # Create composite features
    # Make a feature that contains both longtitude & latitude
    data["diag_coord"] = data["longitude"] + data["latitude"]
    data["bedperroom"] = data["total_bedrooms"] / data["total_rooms"]  # feature w/ bedrooms/room ratio

    # Remove the original features which did not offer value
    del data["total_bedrooms"]
    del data["total_rooms"]

    # TODO: review categorical data
    del data["ocean_proximity"]

    return data


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
    data = process_data(data=data)
    X: pd.DataFrame = data.drop([truth_col_name], axis=1)
    y: pd.DataFrame = data[[truth_col_name]]
    return X, y
