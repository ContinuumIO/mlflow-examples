"""Model Configuration Definition"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from fraud_detection.contracts.dto.abstract import BaseModel


class MLModelConfig(BaseModel):
    """
    Model Config DTO

    Attributes
    ----------
    name: str
        Model (friendly) name.
    model: DecisionTreeClassifier | RandomForestClassifier
        Constructed model (and pipeline).
    params: dict
        pipeline parameters.
    """

    name: str
    model: DecisionTreeClassifier | RandomForestClassifier
    params: dict
