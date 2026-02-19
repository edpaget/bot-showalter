import csv
import fnmatch
import io
import logging
import zipfile
from collections.abc import Callable
from typing import Any

import httpx

from fantasy_baseball_manager.ingest._retry import default_http_retry

logger = logging.getLogger(__name__)

_URL = "https://github.com/chadwickbureau/register/archive/refs/heads/master.zip"

_DEFAULT_RETRY = default_http_retry("Chadwick register download")


class ChadwickRegisterSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "chadwick_bureau"

    @property
    def source_detail(self) -> str:
        return "chadwick_register"

    def _do_fetch(self) -> httpx.Response:
        response = self._client.get(_URL)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("GET %s", _URL)
        response = self._fetch_with_retry()
        logger.debug("Chadwick register responded %d", response.status_code)

        rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            for name in zf.namelist():
                if not fnmatch.fnmatch(name, "*/people-*.csv"):
                    continue
                with zf.open(name) as raw:
                    reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                    for row in reader:
                        if row.get("key_mlbam", "") == "":
                            continue
                        rows.append(dict(row))

        logger.info("Parsed %d player rows from Chadwick register", len(rows))
        return rows
