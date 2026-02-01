import math
from typing import Protocol

import pybaseball


class PlayerIdMapper(Protocol):
    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None: ...
    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None: ...


class ChadwickMapper:
    """Maps player IDs between Yahoo and FanGraphs using the Chadwick register."""

    def __init__(self) -> None:
        register = pybaseball.chadwick_register()
        self._yahoo_to_fg: dict[str, str] = {}
        self._fg_to_yahoo: dict[str, str] = {}

        for _, row in register.iterrows():
            raw_yahoo = row.get("key_yahoo")
            raw_fg = row.get("key_fangraphs")

            if _is_missing(raw_yahoo) or _is_missing(raw_fg):
                continue

            yahoo_id = _to_str_id(raw_yahoo)
            fg_id = _to_str_id(raw_fg)

            self._yahoo_to_fg[yahoo_id] = fg_id
            self._fg_to_yahoo[fg_id] = yahoo_id

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return self._fg_to_yahoo.get(fangraphs_id)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _to_str_id(value: object) -> str:
    if isinstance(value, float):
        return str(int(value))
    return str(value)
