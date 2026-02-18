import csv
import logging
from pathlib import Path
from typing import Any

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

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Reading CSV %s", self._path)
        encoding = params.pop("encoding", "utf-8")
        # Map pandas-style 'sep' to csv.DictReader 'delimiter'
        delimiter = params.pop("sep", params.pop("delimiter", ","))
        with open(self._path, encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows: list[dict[str, Any]] = list(reader)
        logger.debug("Read %d rows from %s", len(rows), self._path)
        return rows
