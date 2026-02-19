import logging
import math
import time
from typing import Any

import requests
from pybaseball import (
    batting_stats_bref,
    fg_batting_data,
    fg_pitching_data,
    pitching_stats_bref,
)
from pylahman import Appearances, People as lahman_people, Teams
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

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

_NETWORK_ERRORS = (requests.RequestException, ConnectionError, TimeoutError)


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning("Retrying pybaseball call (attempt %d): %s", retry_state.attempt_number, retry_state.outcome)


_network_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(_NETWORK_ERRORS),
    before_sleep=_log_retry,
    reraise=True,
)


def _translate_fg_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translate canonical ``season`` kwarg to FanGraphs ``start_season``/``end_season``."""
    if "season" in params:
        yr = params.pop("season")
        params.setdefault("start_season", yr)
        params.setdefault("end_season", yr)
    params.setdefault("qual", 0)
    return params


class FgBattingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "fg_batting_data"

    @_network_retry
    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        translated = _translate_fg_params(params)
        logger.debug("Calling %s(%s)", "fg_batting_data", translated)
        t0 = time.perf_counter()
        try:
            df = fg_batting_data(**translated)
        except _NETWORK_ERRORS:
            raise
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        logger.debug("%s returned %d rows in %.1fs", "fg_batting_data", len(df), time.perf_counter() - t0)
        return df.to_dict("records")


class FgPitchingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "fg_pitching_data"

    @_network_retry
    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        translated = _translate_fg_params(params)
        logger.debug("Calling %s(%s)", "fg_pitching_data", translated)
        t0 = time.perf_counter()
        try:
            df = fg_pitching_data(**translated)
        except _NETWORK_ERRORS:
            raise
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        logger.debug("%s returned %d rows in %.1fs", "fg_pitching_data", len(df), time.perf_counter() - t0)
        return df.to_dict("records")


class BrefBattingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "batting_stats_bref"

    @_network_retry
    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Calling %s(%s)", "batting_stats_bref", params)
        t0 = time.perf_counter()
        try:
            df = batting_stats_bref(**params)
        except _NETWORK_ERRORS:
            raise
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        logger.debug("%s returned %d rows in %.1fs", "batting_stats_bref", len(df), time.perf_counter() - t0)
        return df.to_dict("records")


class BrefPitchingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "pitching_stats_bref"

    @_network_retry
    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Calling %s(%s)", "pitching_stats_bref", params)
        t0 = time.perf_counter()
        try:
            df = pitching_stats_bref(**params)
        except _NETWORK_ERRORS:
            raise
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        logger.debug("%s returned %d rows in %.1fs", "pitching_stats_bref", len(df), time.perf_counter() - t0)
        return df.to_dict("records")


class LahmanPeopleSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "lahman_people"

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Calling %s(%s)", "lahman_people", params)
        t0 = time.perf_counter()
        try:
            df = lahman_people()
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        logger.debug("%s returned %d rows in %.1fs", "lahman_people", len(df), time.perf_counter() - t0)
        return df.to_dict("records")


class LahmanAppearancesSource:
    @property
    def source_type(self) -> str:
        return "pylahman"

    @property
    def source_detail(self) -> str:
        return "appearances"

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Calling %s(%s)", "Appearances", params)
        t0 = time.perf_counter()
        try:
            df = Appearances()
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        if "season" in params:
            df = df[df["yearID"] == params["season"]]
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            for col, pos in _POSITION_COLUMNS.items():
                games = row.get(col, 0)
                if games is not None and not (isinstance(games, float) and math.isnan(games)) and int(games) > 0:
                    records.append(
                        {
                            "playerID": row["playerID"],
                            "yearID": row["yearID"],
                            "teamID": row["teamID"],
                            "position": pos,
                            "games": int(games),
                        }
                    )
        logger.debug("%s returned %d rows in %.1fs", "Appearances", len(records), time.perf_counter() - t0)
        return records


class LahmanTeamsSource:
    @property
    def source_type(self) -> str:
        return "pylahman"

    @property
    def source_detail(self) -> str:
        return "teams"

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Calling %s(%s)", "Teams", params)
        t0 = time.perf_counter()
        try:
            df = Teams()
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        rows = df.to_dict("records")
        if "season" in params:
            rows = [r for r in rows if r.get("yearID") == params["season"]]
        logger.debug("%s returned %d rows in %.1fs", "Teams", len(rows), time.perf_counter() - t0)
        return rows
