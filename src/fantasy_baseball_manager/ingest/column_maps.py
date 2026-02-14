import math
from typing import Any

import pandas as pd

from fantasy_baseball_manager.domain.player import Player


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    int_val = int(value)
    if int_val == -1:
        return None
    return int_val


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value)
    if s == "":
        return None
    return s


def chadwick_row_to_player(row: pd.Series) -> Player | None:
    mlbam_id = _to_optional_int(row["key_mlbam"])
    if mlbam_id is None:
        return None

    return Player(
        name_first=str(row["name_first"]),
        name_last=str(row["name_last"]),
        mlbam_id=mlbam_id,
        fangraphs_id=_to_optional_int(row["key_fangraphs"]),
        bbref_id=_to_optional_str(row["key_bbref"]),
        retro_id=_to_optional_str(row["key_retro"]),
    )
