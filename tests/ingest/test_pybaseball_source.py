from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.ingest.pybaseball_source import (
    BrefBattingSource,
    BrefPitchingSource,
    ChadwickSource,
    FgBattingSource,
    FgPitchingSource,
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
