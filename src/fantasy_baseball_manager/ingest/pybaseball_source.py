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


def _translate_fg_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translate canonical ``season`` kwarg to FanGraphs ``start_season``/``end_season``."""
    if "season" in params:
        yr = params.pop("season")
        params.setdefault("start_season", yr)
        params.setdefault("end_season", yr)
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


class StatcastSource:
    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "statcast"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return statcast(**params)
