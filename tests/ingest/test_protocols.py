from typing import Any

import pandas as pd

from fantasy_baseball_manager.ingest.protocols import DataSource


class FakeSource:
    @property
    def source_type(self) -> str:
        return "test"

    @property
    def source_detail(self) -> str:
        return "fake"

    def fetch(self, **params: Any) -> pd.DataFrame:
        return pd.DataFrame()


class TestDataSourceProtocol:
    def test_fake_source_satisfies_protocol(self) -> None:
        assert isinstance(FakeSource(), DataSource)

    def test_non_conforming_class_fails(self) -> None:
        class NotASource:
            pass

        assert not isinstance(NotASource(), DataSource)
