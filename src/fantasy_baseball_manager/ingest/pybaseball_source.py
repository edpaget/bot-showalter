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
from pylahman import Appearances as lahman_appearances
from pylahman import People as lahman_people

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


def _build_position_column(appearances: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Aggregate Appearances by playerID and return eligible positions.

    Returns a DataFrame with ``playerID`` and ``eligible_positions`` columns.
    Positions with fewer than *min_games* career games are excluded.
    """
    pos_cols = [c for c in _POSITION_COLUMNS if c in appearances.columns]
    grouped = appearances.groupby("playerID")[pos_cols].sum()

    def _to_positions(row: pd.Series) -> str | None:
        pairs = [(int(row[col]), _POSITION_COLUMNS[col]) for col in pos_cols if int(row[col]) >= min_games]
        if not pairs:
            return None
        pairs.sort(key=lambda p: p[0], reverse=True)
        return ",".join(pos for _, pos in pairs)

    positions = grouped.apply(_to_positions, axis=1)
    result = positions.reset_index()
    result.columns = pd.Index(["playerID", "eligible_positions"])
    return result


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
        return chadwick_register()


class FgBattingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "fg_batting_data"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return fg_batting_data(**_translate_fg_params(params))


class FgPitchingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "fg_pitching_data"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return fg_pitching_data(**_translate_fg_params(params))


class BrefBattingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "batting_stats_bref"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return batting_stats_bref(**params)


class BrefPitchingSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "pitching_stats_bref"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return pitching_stats_bref(**params)


class LahmanPeopleSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "lahman_people"

    def fetch(self, **params: Any) -> pd.DataFrame:
        people = lahman_people()
        appearances = lahman_appearances()
        positions = _build_position_column(appearances)
        return people.merge(positions, on="playerID", how="left")


class StatcastSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "statcast"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return statcast(**params)
