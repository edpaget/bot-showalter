import csv
import fnmatch
import io
import logging
import zipfile
from typing import Any

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning(
        "Retrying Chadwick register download (attempt %d): %s", retry_state.attempt_number, retry_state.outcome
    )


_URL = "https://github.com/chadwickbureau/register/archive/refs/heads/master.zip"


class ChadwickRegisterSource:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))

    @property
    def source_type(self) -> str:
        return "chadwick_bureau"

    @property
    def source_detail(self) -> str:
        return "chadwick_register"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _fetch_with_retry(self) -> httpx.Response:
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
