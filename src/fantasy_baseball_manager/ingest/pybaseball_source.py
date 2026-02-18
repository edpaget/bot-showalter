import logging
import time
from typing import Any

import pandas as pd
from pybaseball import (
    batting_stats_bref,
    chadwick_register,
    fg_batting_data,
    fg_pitching_data,
    pitching_stats_bref,
    statcast,
)
from pylahman import Appearances, People as lahman_people, Teams

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


def _translate_fg_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translate canonical ``season`` kwarg to FanGraphs ``start_season``/``end_season``."""
    if "season" in params:
        yr = params.pop("season")
        params.setdefault("start_season", yr)
        params.setdefault("end_season", yr)
    params.setdefault("qual", 0)
    return params


class ChadwickSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "chadwick_register"

    def fetch(self, **params: Any) -> pd.DataFrame:
        logger.debug("Calling %s(%s)", "chadwick_register", params)
        t0 = time.perf_counter()
        df = chadwick_register()
        logger.debug("%s returned %d rows in %.1fs", "chadwick_register", len(df), time.perf_counter() - t0)
        return df


class FgBattingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "fg_batting_data"

    def fetch(self, **params: Any) -> pd.DataFrame:
        translated = _translate_fg_params(params)
        logger.debug("Calling %s(%s)", "fg_batting_data", translated)
        t0 = time.perf_counter()
        df = fg_batting_data(**translated)
        logger.debug("%s returned %d rows in %.1fs", "fg_batting_data", len(df), time.perf_counter() - t0)
        return df


class FgPitchingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "fg_pitching_data"

    def fetch(self, **params: Any) -> pd.DataFrame:
        translated = _translate_fg_params(params)
        logger.debug("Calling %s(%s)", "fg_pitching_data", translated)
        t0 = time.perf_counter()
        df = fg_pitching_data(**translated)
        logger.debug("%s returned %d rows in %.1fs", "fg_pitching_data", len(df), time.perf_counter() - t0)
        return df


class BrefBattingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "batting_stats_bref"

    def fetch(self, **params: Any) -> pd.DataFrame:
        logger.debug("Calling %s(%s)", "batting_stats_bref", params)
        t0 = time.perf_counter()
        df = batting_stats_bref(**params)
        logger.debug("%s returned %d rows in %.1fs", "batting_stats_bref", len(df), time.perf_counter() - t0)
        return df


class BrefPitchingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "pitching_stats_bref"

    def fetch(self, **params: Any) -> pd.DataFrame:
        logger.debug("Calling %s(%s)", "pitching_stats_bref", params)
        t0 = time.perf_counter()
        df = pitching_stats_bref(**params)
        logger.debug("%s returned %d rows in %.1fs", "pitching_stats_bref", len(df), time.perf_counter() - t0)
        return df


class LahmanPeopleSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "lahman_people"

    def fetch(self, **params: Any) -> pd.DataFrame:
        logger.debug("Calling %s(%s)", "lahman_people", params)
        t0 = time.perf_counter()
        df = lahman_people()
        logger.debug("%s returned %d rows in %.1fs", "lahman_people", len(df), time.perf_counter() - t0)
        return df


class StatcastSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "statcast"

    def fetch(self, **params: Any) -> pd.DataFrame:
        logger.debug("Calling %s(%s)", "statcast", params)
        t0 = time.perf_counter()
        df = statcast(**params)
        logger.debug("%s returned %d rows in %.1fs", "statcast", len(df), time.perf_counter() - t0)
        return df


class LahmanAppearancesSource:
    @property
    def source_type(self) -> str:
        return "pylahman"

    @property
    def source_detail(self) -> str:
        return "appearances"

    def fetch(self, **params: Any) -> pd.DataFrame:
        logger.debug("Calling %s(%s)", "Appearances", params)
        t0 = time.perf_counter()
        df = Appearances()
        if "season" in params:
            df = df[df["yearID"] == params["season"]]
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            for col, pos in _POSITION_COLUMNS.items():
                games = row.get(col, 0)
                if pd.notna(games) and int(games) > 0:
                    records.append(
                        {
                            "playerID": row["playerID"],
                            "yearID": row["yearID"],
                            "teamID": row["teamID"],
                            "position": pos,
                            "games": int(games),
                        }
                    )
        result = pd.DataFrame(records)
        logger.debug("%s returned %d rows in %.1fs", "Appearances", len(result), time.perf_counter() - t0)
        return result


class LahmanTeamsSource:
    @property
    def source_type(self) -> str:
        return "pylahman"

    @property
    def source_detail(self) -> str:
        return "teams"

    def fetch(self, **params: Any) -> pd.DataFrame:
        logger.debug("Calling %s(%s)", "Teams", params)
        t0 = time.perf_counter()
        df = Teams()
        if "season" in params:
            df = df[df["yearID"] == params["season"]]
        logger.debug("%s returned %d rows in %.1fs", "Teams", len(df), time.perf_counter() - t0)
        return df
