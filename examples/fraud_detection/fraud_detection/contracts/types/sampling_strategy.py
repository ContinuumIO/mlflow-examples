""" Enumerations for supported sampling strategies. """

from enum import Enum


class SamplingStrategyType(str, Enum):
    """Sampling Strategy Type Enumeration"""

    UNDERSAMPLINGNEARMISS = "under_sampling_near_miss"
    OVERSAMPLINGSMOTE = "over_sampling_smote"
