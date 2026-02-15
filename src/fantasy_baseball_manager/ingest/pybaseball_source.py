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
        return lahman_people()


class StatcastSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "statcast"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return statcast(**params)


class LahmanAppearancesSource:
    @property
    def source_type(self) -> str:
        return "pylahman"

    @property
    def source_detail(self) -> str:
        return "appearances"

    def fetch(self, **params: Any) -> pd.DataFrame:
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
        return pd.DataFrame(records)


class LahmanTeamsSource:
    @property
    def source_type(self) -> str:
        return "pylahman"

    @property
    def source_detail(self) -> str:
        return "teams"

    def fetch(self, **params: Any) -> pd.DataFrame:
        df = Teams()
        if "season" in params:
            df = df[df["yearID"] == params["season"]]
        return df
