"""This module contains the Model Factory Definition"""

from __future__ import annotations

# Imbalanced Learning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Preprocessing
from sklearn.preprocessing import RobustScaler

# ML Models
from sklearn.tree import DecisionTreeClassifier

from fraud_detection.common.constants import rng
from fraud_detection.contracts.dto.abstract import BaseModel
from fraud_detection.contracts.dto.ml_model_config import MLModelConfig
from fraud_detection.contracts.types.model import ModelType
from fraud_detection.contracts.types.sampling_strategy import SamplingStrategyType


class ModelFactory(BaseModel):
    """
    Model Factory
    This class is responsible for creating model configs.
    """

    model_type: ModelType
    strategy: SamplingStrategyType

    #############################################################################
    # Model Methods
    #############################################################################

    def build_model(self) -> MLModelConfig:
        """
        Generates and returns a model config (based on the instance of the factory).

        Returns
        -------
        ml_model_config: MLModelConfig
            The model config for the requested model.
        """

        if self.model_type == ModelType.DECISIONTREE:
            return ModelFactory._build_decision_tree()
        if self.model_type == ModelType.RANDOMFOREST:
            return self._build_random_forest()

        message: str = f"Unknown model type provided: {self.model_type}"
        raise NotImplementedError(message)

    @staticmethod
    def _build_decision_tree() -> MLModelConfig:
        """
        Generates and returns an instance of a decision tree model config.

        Returns
        -------
        dt_config: MLModelConfig
            A decision tree model config instance.
        """

        # Decision Tree
        decision_tree_classifier: DecisionTreeClassifier = DecisionTreeClassifier(random_state=rng)
        dt_params: dict = ModelFactory._build_common_tree_params()
        return MLModelConfig(name="Decision Tree", model=decision_tree_classifier, params=dt_params)

    def _build_random_forest(self) -> MLModelConfig:
        """
        Generates and returns an instance of a random forest model config.

        Returns
        -------
        rf_config: MLModelConfig
            A random forest model config instance.
        """

        # Random Forest

        random_forest_classifier: RandomForestClassifier = RandomForestClassifier(random_state=rng, n_jobs=-1)
        rf_params: dict = {
            "model__n_estimators": [
                50,
            ],
            "model__max_features": ["log2", "sqrt"],
        }

        # only RF specific params tuned
        if self.strategy == SamplingStrategyType.UNDERSAMPLINGNEARMISS:
            rf_params = ModelFactory._build_common_tree_params() | rf_params

        return MLModelConfig(name="Random Forest", model=random_forest_classifier, params=rf_params)

    #############################################################################
    # Preprocessing Methods
    #############################################################################

    def build_model_preprocessing_steps(self) -> dict:
        """
        Generates a dictionary of model preprocessing steps suitable for use in the instantiation of a `TrainingParams`.

        Returns
        -------
        model_config_partial: dict
            Returns a dictionary with the `preprocessing_steps` and `preprocessing_params`
             key/values suitable for use in the instantiation of a `TrainingParams`.
        """

        preproc_steps, preproc_params = self._build_preprocessing_steps()
        model_config_partial: dict = {
            "preprocessing_steps": preproc_steps,
        }
        if preproc_params:
            model_config_partial["preprocessing_params"] = preproc_params
        return model_config_partial

    def _build_preprocessing_steps(self) -> tuple[list[tuple[str, ColumnTransformer], tuple[str, NearMiss | SMOTE]], dict | None]:
        """
        Generate common preprocessing stanza.

        Returns
        -------
        preproc_steps: tuple, prepoc_params: dict | None
            Preprocessing pipeline steps
        """

        # (Selected) Feature Scaling
        preprocessing = ColumnTransformer(
            [
                ("scaler", RobustScaler(), ["Time", "Amount"]),
            ],
            remainder="passthrough",
        )

        # Build sampling strategy and hyperparameters (if any).
        strat, strat_params = self._build_sampling_strategy()

        return [("preprocess", preprocessing), ("sampling", strat)], strat_params

    @staticmethod
    def _build_common_tree_params() -> dict:
        """
        Generates and returns a diction of common tree parameters.

        Returns
        -------
        common_params: dict
            A diction of common tree paramters.
        """

        return {
            "model__max_depth": [None, 2, 3, 6],
            "model__min_samples_leaf": [2, 5, 6],
            "model__criterion": ["gini", "entropy"],
        }

    #############################################################################
    # Sampling Strategy Methods
    #############################################################################

    def _build_sampling_strategy(self) -> tuple[NearMiss | SMOTE, dict | None]:
        """
        Builds the sampling strategy for use in the pipeline.

        Returns
        -------
        strat, strat_params
            An instance of the sampling strategy, and optionally any hyperparameters it requires.
        """

        if self.strategy == SamplingStrategyType.UNDERSAMPLINGNEARMISS:
            return ModelFactory._build_near_miss_strategy()
        if self.strategy == SamplingStrategyType.OVERSAMPLINGSMOTE:
            return ModelFactory._build_smote_strategy()

        message: str = f"Unknown strategy provided: {self.strategy}"
        raise NotImplementedError(message)

    @staticmethod
    def _build_near_miss_strategy() -> tuple[NearMiss, dict]:
        """
        Builds the NearMiss sampling strategy.

        Returns
        -------
        strategy: tuple[NearMiss, dict]
            An instance of NearMiss and its associated hyperparameters.
        """

        # Under Sampling Strategy
        near_miss: NearMiss = NearMiss(sampling_strategy="majority", version=3)

        # NearMiss Param Grid
        nm_params: dict = {"sampling__n_neighbors_ver3": [4, 5]}

        return near_miss, nm_params

    @staticmethod
    def _build_smote_strategy() -> tuple[SMOTE, None]:
        """
        Builds the SMOTE sampling strategy.

        Returns
        -------
        strategy: tuple[SMOTE, None]
            An instance of SMOTE.
        """

        # Over Sampling Strategies
        smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=rng)
        return smote, None
