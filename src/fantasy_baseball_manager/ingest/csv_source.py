import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class CsvSource:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def source_type(self) -> str:
        return "csv"

    @property
    def source_detail(self) -> str:
        return str(self._path)

    def fetch(self, **params: Any) -> pd.DataFrame:
        logger.debug("Reading CSV %s", self._path)
        df = pd.read_csv(self._path, **params)
        logger.debug("Read %d rows from %s", len(df), self._path)
        return df
