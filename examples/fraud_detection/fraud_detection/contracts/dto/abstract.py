"""
This module holds the `BaseModel` definition which we subclass from Pydantic in order to apply our business logic.
"""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    """BaseModel DTO"""

    # Pydantic class config override
    class Config:
        arbitrary_types_allowed = True
