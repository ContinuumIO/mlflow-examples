""" Dashboard Client Data Response Definition """

from __future__ import annotations

from fraud_detection.contracts.dto.abstract import BaseModel


class DashboardClientResponse(BaseModel):
    """
    DashboardClientResponse DTO

    Attributes
    ----------
    time: float
        Time stamp for the sample.
    amount: float
        The transaction amount.
    label: int | None = None
        Optional true label (if available)
    prediction: int
        The predicted class/label
    features: list | None = None
        Optional list of features for the sample.
    """

    time: float
    amount: float
    label: int | None = None
    prediction: int
    features: list | None = None
