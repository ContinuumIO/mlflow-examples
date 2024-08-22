"""
###############################################################################
# Data Streaming Service REST API
###############################################################################
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import Any

from fastapi import FastAPI
from starlette.exceptions import HTTPException

from fraud_detection.contracts.dto.stream_status import StreamStatus
from fraud_detection.services.data import DataService

###############################################################################
# Application Setup
###############################################################################

logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)

# Create our data service.
# pylint: disable=broad-exception-caught
try:
    data_service: DataService = DataService()
except Exception as error:
    logger.error(str(error))

# Create the FastAPI App.
app: FastAPI = FastAPI()


###############################################################################
# Wrappers
###############################################################################


def wrapped_function_call(func, *args, **kwargs) -> Any:
    """
    Handles exception handling when calling functions.

    Parameters
    ----------
    func
        Function to call
    args
        args
    kwargs
        kwargs

    Returns
    -------
    any: Any
        Function return
    """

    # pylint: disable=broad-exception-caught
    try:
        return func(*args, **kwargs)
    except Exception as local_error:
        if isinstance(local_error, HTTPException):
            raise local_error
        raise HTTPException(status_code=500, detail=str(local_error)) from local_error


###############################################################################
# Handlers
###############################################################################


@app.get("/api/v1/sample", status_code=200)
async def stream_sample_data() -> dict:
    """Returns a single sample from the data stream."""
    return wrapped_function_call(func=data_service.get)


@app.put("/api/v1/reset", status_code=201)
async def stream_reset() -> None:
    """Resets the Stream API time series index."""
    return wrapped_function_call(func=data_service.reset)


@app.get("/api/v1/index", status_code=200)
async def stream_index(value: int) -> dict:
    """Returns the data at the requested time series index."""
    return wrapped_function_call(func=data_service.index, index=value)


@app.get("/api/v1/batch", status_code=200)
async def stream_batch(start: int, size: int) -> list[dict]:
    """Generates a batch of data. Starting at `start` index and of size `size`."""
    return wrapped_function_call(func=data_service.batch, start=start, size=size)


@app.get("/api/v1/info", status_code=200)
async def stream_info() -> StreamStatus:
    """Get Data Stream API status"""
    return wrapped_function_call(func=data_service.info())
