""" Defines a data sample from the Data Stream Service. """

from __future__ import annotations

import pandas as pd

from fraud_detection.contracts.dto.abstract import BaseModel


class StreamData(BaseModel):
    """
    StreamData DTO

    Attributes
    ----------
    X: pd.DataFrame
        X (input data)
    true_label: int | None = None
        True label of the input data (if available).
    """

    X: pd.DataFrame
    true_label: int | None = None
