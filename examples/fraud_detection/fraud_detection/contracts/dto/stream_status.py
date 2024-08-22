"""Defines the Data Steam API Status"""

from __future__ import annotations

from fraud_detection.contracts.dto.abstract import BaseModel


class StreamStatus(BaseModel):
    """
    StreamStatus DTO

    Attributes
    ----------
    counter: int
        Current time series index value.
    max_batch_size: int
        The latest batch size which a user can request.
    store_size: int
        The data set size.
    """

    counter: int
    max_batch_size: int
    store_size: int
