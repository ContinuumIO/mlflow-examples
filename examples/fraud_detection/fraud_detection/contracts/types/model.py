""" Enumerations for supported model types. """

from enum import Enum


class ModelType(str, Enum):
    """ModelType Enumeration"""

    DECISIONTREE = "decision_tree"
    RANDOMFOREST = "random_forest"
