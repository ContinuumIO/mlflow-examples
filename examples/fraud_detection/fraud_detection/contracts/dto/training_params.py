""" Training Parameters Definition """

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.compose import ColumnTransformer

from fraud_detection.contracts.dto.abstract import BaseModel
from fraud_detection.contracts.dto.dataset import DataSet
from fraud_detection.contracts.dto.ml_model_config import MLModelConfig


class TrainingParams(BaseModel):
    """
    TrainingParams DTO

    Attributes
    ----------
    ml_model_config: MLModelConfig
        Model definition.
    preprocessing_steps: list[tuple[str, ColumnTransformer | NearMiss | SMOTE]]
        Preprocessing pipeline for the model training.
    preprocessing_params: dict | None = None
        Optional preprocessing pipeline hyperparameters.
    dataset: DataSet
        The dataset to train over.
    cv_n_reps: int
        Number of times cross-validator needs to be repeated.
    cv_n_splits: int
        Number of folds. Must be at least 2.
    """

    ml_model_config: MLModelConfig
    preprocessing_steps: list[tuple[str, ColumnTransformer | NearMiss | SMOTE]]
    preprocessing_params: dict | None = None
    dataset: DataSet
    cv_n_reps: int = 10
    cv_n_splits: int = 5
