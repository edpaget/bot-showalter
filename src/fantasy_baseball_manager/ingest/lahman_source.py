import csv
import io
import logging
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from fantasy_baseball_manager.ingest._csv_helpers import nullify_empty_strings, strip_bom

logger = logging.getLogger(__name__)

_BASE_URL = "https://raw.githubusercontent.com/daviddalpiaz/pylahman/main/data-raw"
_PEOPLE_URL = f"{_BASE_URL}/People.csv"
_APPEARANCES_URL = f"{_BASE_URL}/Appearances.csv"
_TEAMS_URL = f"{_BASE_URL}/Teams.csv"

_POSITION_COLUMNS: dict[str, str] = {
    "G_p": "P",
    "G_c": "C",
    "G_1b": "1B",
    "G_2b": "2B",
    "G_3b": "3B",
    "G_ss": "SS",
    "G_lf": "LF",
    "G_cf": "CF",
    "G_rf": "RF",
    "G_dh": "DH",
}


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning("Retrying Lahman download (attempt %d): %s", retry_state.attempt_number, retry_state.outcome)


_DEFAULT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    before_sleep=_log_retry,
    reraise=True,
)


class LahmanPeopleSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "lahman"

    @property
    def source_detail(self) -> str:
        return "people"

    def _do_fetch(self) -> httpx.Response:
        response = self._client.get(_PEOPLE_URL)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("GET %s", _PEOPLE_URL)
        response = self._fetch_with_retry()
        reader = csv.DictReader(io.StringIO(strip_bom(response.text)))
        rows: list[dict[str, Any]] = [nullify_empty_strings(row) for row in reader]
        logger.info("Parsed %d Lahman People rows", len(rows))
        return rows


class LahmanAppearancesSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "lahman"

    @property
    def source_detail(self) -> str:
        return "appearances"

    def _do_fetch(self) -> httpx.Response:
        response = self._client.get(_APPEARANCES_URL)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("GET %s", _APPEARANCES_URL)
        response = self._fetch_with_retry()
        reader = csv.DictReader(io.StringIO(strip_bom(response.text)))
        all_rows = [nullify_empty_strings(row) for row in reader]

        season = params.get("season")
        if season is not None:
            all_rows = [r for r in all_rows if r["yearID"] == str(season)]

        records: list[dict[str, Any]] = []
        for row in all_rows:
            for col, pos in _POSITION_COLUMNS.items():
                games_val = row.get(col)
                if games_val is None:
                    continue
                games = int(games_val)
                if games > 0:
                    records.append(
                        {
                            "playerID": row["playerID"],
                            "yearID": int(row["yearID"]),
                            "teamID": row["teamID"],
                            "position": pos,
                            "games": games,
                        }
                    )

        logger.info("Parsed %d Lahman Appearances records", len(records))
        return records


class LahmanTeamsSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "lahman"

    @property
    def source_detail(self) -> str:
        return "teams"

    def _do_fetch(self) -> httpx.Response:
        response = self._client.get(_TEAMS_URL)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("GET %s", _TEAMS_URL)
        response = self._fetch_with_retry()
        reader = csv.DictReader(io.StringIO(strip_bom(response.text)))
        rows: list[dict[str, Any]] = [nullify_empty_strings(row) for row in reader]

        season = params.get("season")
        if season is not None:
            rows = [r for r in rows if r["yearID"] == str(season)]

        logger.info("Parsed %d Lahman Teams rows", len(rows))
        return rows
