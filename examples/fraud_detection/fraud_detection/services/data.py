"""Data Service Definition"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ae5_tools import demand_env_var
from starlette.exceptions import HTTPException

from fraud_detection.contracts.dto.stream_status import StreamStatus


class DataService:
    """Data Stream API Internal Data Service"""

    def __init__(self):
        self.access_counter: int = 1
        self.max_batch_size: int = 1000

        source_data_path: Path = Path(demand_env_var(name="DATA_BASE_DIR")) / demand_env_var(name="DATA_ARTIFACT")
        self.data_store: pd.DataFrame = pd.read_csv(source_data_path)
        self.store_size: int = self.data_store.shape[0]

    def get(self) -> dict:
        """
        Returns a single sample from the data stream.

        Returns
        -------
        sample: dict
            A dictionary (entity data from the stream)
        """

        local_data: pd.DataFrame = self.data_store.iloc[[self.access_counter - 1]]

        if self.access_counter < self.store_size:
            self.access_counter += 1
        else:
            # loop around ..
            self.access_counter = 1

        return local_data.to_dict(orient="records")[0]

    def batch(self, start: int, size: int) -> list:
        """
        Generates a batch of data. Starting at `start` index and of size `size`.

        Parameters
        ----------
        start: int
            Start index
        size: int
            Batch size

        Returns
        -------
        batch: list[dict]
            A list of dictionaries (data from entities from the stream).
        """

        if size < 1:
            raise HTTPException(status_code=400, detail="size too small")
        if start > self.store_size:
            raise HTTPException(status_code=400, detail="start too large")

        if size > self.max_batch_size:
            size = self.max_batch_size

        if (start + size) >= self.store_size:
            local_data = self.data_store[(start - 1) : self.store_size]
        else:
            local_data = self.data_store[(start - 1) : ((start - 1) + size)]

        return local_data.to_dict(orient="records")

    def reset(self) -> None:
        """Resets the Stream API time series index."""
        self.access_counter = 1

    def index(self, index: int) -> dict:
        """
        Returns the data at the requested time series index.

        Returns
        -------
        sample: dict
            A dictionary (entity data from the stream)
        """

        return self.data_store.iloc[[index - 1]].to_dict(orient="records")[0]

    def info(self) -> StreamStatus:
        """
        Get Data Stream API status

        Returns
        -------
        status: StreamStatus
            Data Stream API Status
        """

        return StreamStatus(counter=self.access_counter, max_batch_size=self.max_batch_size, store_size=self.store_size)
