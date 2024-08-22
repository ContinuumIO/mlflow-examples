"""Dashboard MLflow Reports Definition"""

import pandas as pd

from fraud_detection.contracts.dto.abstract import BaseModel


class DashboardMLflowReport(BaseModel):
    """
    DashboardMLflowReport DTO

    Attributes
    ----------
    workflow: pd.DataFrame
        Training workflow report.
    model: pd.DataFrame
        Model registry report.
    """

    workflow: pd.DataFrame
    model: pd.DataFrame
