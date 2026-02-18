import pytest
import requests

from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.ingest.pybaseball_source import (
    BrefBattingSource,
    BrefPitchingSource,
    ChadwickSource,
    FgBattingSource,
    FgPitchingSource,
    LahmanAppearancesSource,
    LahmanPeopleSource,
    LahmanTeamsSource,
    StatcastSource,
    _translate_fg_params,
)


class TestChadwickSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(ChadwickSource(), DataSource)

    def test_source_type(self) -> None:
        assert ChadwickSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert ChadwickSource().source_detail == "chadwick_register"


class TestFgBattingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(FgBattingSource(), DataSource)

    def test_source_type(self) -> None:
        assert FgBattingSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert FgBattingSource().source_detail == "fg_batting_data"


class TestFgPitchingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(FgPitchingSource(), DataSource)

    def test_source_type(self) -> None:
        assert FgPitchingSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert FgPitchingSource().source_detail == "fg_pitching_data"


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


class TestTranslateFgParams:
    def test_translates_season_to_start_end(self) -> None:
        result = _translate_fg_params({"season": 2023})
        assert result == {"start_season": 2023, "end_season": 2023, "qual": 0}

    def test_passthrough_when_no_season(self) -> None:
        result = _translate_fg_params({"start_season": 2022, "end_season": 2023})
        assert result == {"start_season": 2022, "end_season": 2023, "qual": 0}

    def test_does_not_override_explicit_start_end(self) -> None:
        result = _translate_fg_params({"season": 2023, "start_season": 2020})
        assert result == {"start_season": 2020, "end_season": 2023, "qual": 0}

    def test_defaults_qual_to_zero(self) -> None:
        result = _translate_fg_params({})
        assert result["qual"] == 0

    def test_does_not_override_explicit_qual(self) -> None:
        result = _translate_fg_params({"qual": 100})
        assert result["qual"] == 100


class TestLahmanPeopleSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(LahmanPeopleSource(), DataSource)

    def test_source_type(self) -> None:
        assert LahmanPeopleSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert LahmanPeopleSource().source_detail == "lahman_people"


class TestStatcastSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(StatcastSource(), DataSource)

    def test_source_type(self) -> None:
        assert StatcastSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert StatcastSource().source_detail == "statcast"


class TestLahmanAppearancesSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(LahmanAppearancesSource(), DataSource)

    def test_source_type(self) -> None:
        assert LahmanAppearancesSource().source_type == "pylahman"

    def test_source_detail(self) -> None:
        assert LahmanAppearancesSource().source_detail == "appearances"


class TestLahmanTeamsSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(LahmanTeamsSource(), DataSource)

    def test_source_type(self) -> None:
        assert LahmanTeamsSource().source_type == "pylahman"

    def test_source_detail(self) -> None:
        assert LahmanTeamsSource().source_detail == "teams"


class TestErrorWrapping:
    def test_non_network_error_wrapped_as_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(**kw):  # type: ignore[no-untyped-def]
            raise ValueError("bad data")

        monkeypatch.setattr(
            "fantasy_baseball_manager.ingest.pybaseball_source.fg_batting_data",
            fail,
        )
        source = FgBattingSource()
        with pytest.raises(RuntimeError, match="pybaseball fetch failed"):
            source.fetch(season=2024)

    def test_network_error_retried_then_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def fail(**kw):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr(
            "fantasy_baseball_manager.ingest.pybaseball_source.fg_batting_data",
            fail,
        )
        source = FgBattingSource()
        with pytest.raises(requests.ConnectionError):
            source.fetch(season=2024)

        assert call_count == 2
