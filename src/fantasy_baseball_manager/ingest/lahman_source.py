import csv
import io
import logging
from collections.abc import Callable
from typing import Any

import httpx

from fantasy_baseball_manager.ingest._csv_helpers import nullify_empty_strings, strip_bom
from fantasy_baseball_manager.ingest._retry import default_http_retry

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

_DEFAULT_RETRY = default_http_retry("Lahman download")


class LahmanCsvSource:
    _url: str
    _source_detail: str

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
        return self._source_detail

    def _do_fetch(self) -> httpx.Response:
        response = self._client.get(self._url)
        response.raise_for_status()
        return response

    def _post_process(self, rows: list[dict[str, Any]], **params: Any) -> list[dict[str, Any]]:
        return rows

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("GET %s", self._url)
        response = self._fetch_with_retry()
        reader = csv.DictReader(io.StringIO(strip_bom(response.text)))
        rows: list[dict[str, Any]] = [nullify_empty_strings(row) for row in reader]
        rows = self._post_process(rows, **params)
        logger.info("Parsed %d Lahman %s rows", len(rows), self._source_detail.title())
        return rows


class LahmanPeopleSource(LahmanCsvSource):
    _url = _PEOPLE_URL
    _source_detail = "people"


class LahmanAppearancesSource(LahmanCsvSource):
    _url = _APPEARANCES_URL
    _source_detail = "appearances"

    def _post_process(self, rows: list[dict[str, Any]], **params: Any) -> list[dict[str, Any]]:
        season = params.get("season")
        if season is not None:
            rows = [r for r in rows if r["yearID"] == str(season)]

        records: list[dict[str, Any]] = []
        for row in rows:
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

        return records


class LahmanTeamsSource(LahmanCsvSource):
    _url = _TEAMS_URL
    _source_detail = "teams"

    def _post_process(self, rows: list[dict[str, Any]], **params: Any) -> list[dict[str, Any]]:
        season = params.get("season")
        if season is not None:
            rows = [r for r in rows if r["yearID"] == str(season)]
        return rows
