from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball_manager.statcast.models import SeasonManifest
from fantasy_baseball_manager.statcast.store import StatcastStore


@pytest.fixture
def store(tmp_path: Path) -> StatcastStore:
    return StatcastStore(data_dir=tmp_path)


class TestManifest:
    def test_load_returns_empty_manifest_when_none_exists(self, store: StatcastStore) -> None:
        manifest = store.load_manifest(2024)
        assert manifest.season == 2024
        assert manifest.fetched_dates == set()
        assert manifest.total_rows == 0

    def test_save_and_load_roundtrip(self, store: StatcastStore) -> None:
        manifest = SeasonManifest(
            season=2024,
            fetched_dates={"2024-03-20", "2024-03-21"},
            total_rows=150,
        )
        store.save_manifest(manifest)
        loaded = store.load_manifest(2024)
        assert loaded.season == 2024
        assert loaded.fetched_dates == {"2024-03-20", "2024-03-21"}
        assert loaded.total_rows == 150

    def test_save_creates_season_directory(self, store: StatcastStore) -> None:
        manifest = SeasonManifest(season=2024)
        store.save_manifest(manifest)
        season_dir = store._data_dir / "2024"
        assert season_dir.is_dir()
        assert (season_dir / "manifest.json").exists()


class TestFlushMonth:
    def test_writes_parquet_file(self, store: StatcastStore) -> None:
        df = pd.DataFrame({"pitch_type": ["FF", "SL"], "release_speed": [95.0, 85.0]})
        store.flush_month(2024, 4, df)
        path = store._data_dir / "2024" / "statcast_2024_04.parquet"
        assert path.exists()

    def test_appends_to_existing_parquet(self, store: StatcastStore) -> None:
        df1 = pd.DataFrame({"pitch_type": ["FF"], "release_speed": [95.0]})
        df2 = pd.DataFrame({"pitch_type": ["SL"], "release_speed": [85.0]})
        store.flush_month(2024, 4, df1)
        store.flush_month(2024, 4, df2)
        result = store.read_month(2024, 4)
        assert result is not None
        assert len(result) == 2


class TestReadMonth:
    def test_returns_none_when_no_file(self, store: StatcastStore) -> None:
        result = store.read_month(2024, 4)
        assert result is None

    def test_returns_dataframe_when_exists(self, store: StatcastStore) -> None:
        df = pd.DataFrame({"pitch_type": ["FF", "SL"], "release_speed": [95.0, 85.0]})
        store.flush_month(2024, 4, df)
        result = store.read_month(2024, 4)
        assert result is not None
        assert len(result) == 2
        assert list(result.columns) == ["pitch_type", "release_speed"]


class TestReadSeason:
    def test_returns_empty_dataframe_when_no_files(self, store: StatcastStore) -> None:
        result = store.read_season(2024)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_concatenates_multiple_months(self, store: StatcastStore) -> None:
        df_apr = pd.DataFrame({"pitch_type": ["FF"], "release_speed": [95.0]})
        df_may = pd.DataFrame({"pitch_type": ["SL"], "release_speed": [85.0]})
        store.flush_month(2024, 4, df_apr)
        store.flush_month(2024, 5, df_may)
        result = store.read_season(2024)
        assert len(result) == 2
