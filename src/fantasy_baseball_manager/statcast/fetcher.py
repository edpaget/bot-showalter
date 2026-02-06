from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


@runtime_checkable
class StatcastFetcher(Protocol):
    def fetch_day(self, day: date) -> pd.DataFrame: ...


class PybaseballFetcher:
    def fetch_day(self, day: date) -> pd.DataFrame:
        import pybaseball

        date_str = day.strftime("%Y-%m-%d")
        return pybaseball.statcast(start_dt=date_str, end_dt=date_str, verbose=False)
