from pathlib import Path
from typing import Any

import pandas as pd


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
        return pd.read_csv(self._path, **params)
