from datetime import date

import pandas as pd

from fantasy_baseball_manager.statcast.fetcher import PybaseballFetcher, StatcastFetcher


class FakeFetcher:
    """Stub that satisfies StatcastFetcher protocol."""

    def __init__(self, data: dict[date, pd.DataFrame] | None = None) -> None:
        self._data = data or {}

    def fetch_day(self, day: date) -> pd.DataFrame:
        return self._data.get(day, pd.DataFrame())


class TestStatcastFetcherProtocol:
    def test_fake_satisfies_protocol(self) -> None:
        fetcher: StatcastFetcher = FakeFetcher()
        result = fetcher.fetch_day(date(2024, 4, 1))
        assert isinstance(result, pd.DataFrame)

    def test_fake_returns_canned_data(self) -> None:
        df = pd.DataFrame({"pitch_type": ["FF", "SL"]})
        fetcher = FakeFetcher(data={date(2024, 4, 1): df})
        result = fetcher.fetch_day(date(2024, 4, 1))
        assert len(result) == 2

    def test_fake_returns_empty_for_unknown_date(self) -> None:
        fetcher = FakeFetcher()
        result = fetcher.fetch_day(date(2024, 4, 1))
        assert len(result) == 0


class TestPybaseballFetcher:
    def test_is_protocol_compatible(self) -> None:
        fetcher: StatcastFetcher = PybaseballFetcher()
        assert hasattr(fetcher, "fetch_day")
