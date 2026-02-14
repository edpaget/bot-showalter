from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.ingest.pybaseball_source import ChadwickSource


class TestChadwickSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(ChadwickSource(), DataSource)

    def test_source_type(self) -> None:
        assert ChadwickSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert ChadwickSource().source_detail == "chadwick_register"
