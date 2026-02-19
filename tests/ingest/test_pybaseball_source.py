import pytest
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.ingest.pybaseball_source import (
    BrefBattingSource,
    BrefPitchingSource,
)

_NO_WAIT_RETRY = retry(
    stop=stop_after_attempt(2),
    wait=wait_none(),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
    reraise=True,
)


class TestBrefBattingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(BrefBattingSource(), DataSource)

    def test_source_type(self) -> None:
        assert BrefBattingSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert BrefBattingSource().source_detail == "batting_stats_bref"


class TestBrefPitchingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(BrefPitchingSource(), DataSource)

    def test_source_type(self) -> None:
        assert BrefPitchingSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert BrefPitchingSource().source_detail == "pitching_stats_bref"


class TestErrorWrapping:
    def test_non_network_error_wrapped_as_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(**kw):
            raise ValueError("bad data")

        monkeypatch.setattr(
            "fantasy_baseball_manager.ingest.pybaseball_source.batting_stats_bref",
            fail,
        )
        source = BrefBattingSource()
        with pytest.raises(RuntimeError, match="pybaseball fetch failed"):
            source.fetch(season=2024)

    def test_network_error_retried_then_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def fail(**kw):
            nonlocal call_count
            call_count += 1
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr(
            "fantasy_baseball_manager.ingest.pybaseball_source.batting_stats_bref",
            fail,
        )
        source = BrefBattingSource(retry=_NO_WAIT_RETRY)
        with pytest.raises(requests.ConnectionError):
            source.fetch(season=2024)

        assert call_count == 2
