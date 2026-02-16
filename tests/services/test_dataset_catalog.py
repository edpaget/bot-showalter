import json
import sqlite3
from collections.abc import Generator

import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.services.dataset_catalog import DatasetCatalogService, DatasetInfo


def _seed_feature_set(conn: sqlite3.Connection, name: str, version: str) -> int:
    cursor = conn.execute(
        "INSERT INTO feature_set (name, version, created_at) VALUES (?, ?, '2025-01-01T00:00:00')",
        (name, version),
    )
    return cursor.lastrowid  # type: ignore[return-value]


def _seed_dataset(
    conn: sqlite3.Connection,
    feature_set_id: int,
    *,
    split: str | None = None,
    table_name: str | None = None,
    row_count: int = 10,
    seasons: list[int] | None = None,
) -> int:
    seasons_json = json.dumps(seasons or [2024])
    cursor = conn.execute(
        "INSERT INTO dataset (feature_set_id, name, split, table_name, row_count, seasons, created_at) "
        "VALUES (?, 'ds', ?, ?, ?, ?, '2025-01-01T00:00:00')",
        (feature_set_id, split, table_name, row_count, seasons_json),
    )
    dataset_id: int = cursor.lastrowid  # type: ignore[assignment]
    if table_name is None:
        table_name = f"ds_{dataset_id}"
        conn.execute(
            "UPDATE dataset SET table_name = ? WHERE id = ?",
            (table_name, dataset_id),
        )
    # Create the actual table
    conn.execute(f"CREATE TABLE IF NOT EXISTS [{table_name}] (x INTEGER)")
    conn.commit()
    return dataset_id


@pytest.fixture
def conn() -> Generator[sqlite3.Connection]:
    connection = create_connection(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def catalog(conn: sqlite3.Connection) -> DatasetCatalogService:
    return DatasetCatalogService(conn)


class TestListAll:
    def test_empty(self, catalog: DatasetCatalogService) -> None:
        assert catalog.list_all() == []

    def test_returns_all_datasets(self, conn: sqlite3.Connection, catalog: DatasetCatalogService) -> None:
        fs_id = _seed_feature_set(conn, "batting_features", "abc123")
        _seed_dataset(conn, fs_id, seasons=[2023, 2024])

        fs_id2 = _seed_feature_set(conn, "pitching_features", "def456")
        _seed_dataset(conn, fs_id2, seasons=[2024])

        result = catalog.list_all()
        assert len(result) == 2
        assert all(isinstance(r, DatasetInfo) for r in result)
        names = {r.feature_set_name for r in result}
        assert names == {"batting_features", "pitching_features"}

    def test_includes_split_datasets(self, conn: sqlite3.Connection, catalog: DatasetCatalogService) -> None:
        fs_id = _seed_feature_set(conn, "batting_features", "abc123")
        _seed_dataset(conn, fs_id)
        _seed_dataset(conn, fs_id, split="train", table_name="ds_1_train")
        _seed_dataset(conn, fs_id, split="val", table_name="ds_1_val")

        result = catalog.list_all()
        assert len(result) == 3


class TestListByFeatureSetName:
    def test_filters_by_name(self, conn: sqlite3.Connection, catalog: DatasetCatalogService) -> None:
        fs1 = _seed_feature_set(conn, "batting_features", "abc123")
        _seed_dataset(conn, fs1)

        fs2 = _seed_feature_set(conn, "pitching_features", "def456")
        _seed_dataset(conn, fs2)

        result = catalog.list_by_feature_set_name("batting_features")
        assert len(result) == 1
        assert result[0].feature_set_name == "batting_features"

    def test_no_match(self, catalog: DatasetCatalogService) -> None:
        assert catalog.list_by_feature_set_name("nonexistent") == []


class TestDropByFeatureSetName:
    def test_drops_tables_and_metadata(self, conn: sqlite3.Connection, catalog: DatasetCatalogService) -> None:
        fs_id = _seed_feature_set(conn, "batting_features", "abc123")
        ds_id = _seed_dataset(conn, fs_id)
        table_name = f"ds_{ds_id}"

        # Verify table exists
        assert _table_exists(conn, table_name)

        count = catalog.drop_by_feature_set_name("batting_features")
        assert count == 1

        # Table should be gone
        assert not _table_exists(conn, table_name)

        # Metadata should be gone
        assert conn.execute("SELECT COUNT(*) FROM dataset WHERE feature_set_id = ?", (fs_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM feature_set WHERE id = ?", (fs_id,)).fetchone()[0] == 0

    def test_drops_split_tables_too(self, conn: sqlite3.Connection, catalog: DatasetCatalogService) -> None:
        fs_id = _seed_feature_set(conn, "batting_features", "abc123")
        _seed_dataset(conn, fs_id)
        _seed_dataset(conn, fs_id, split="train", table_name="ds_1_train")
        _seed_dataset(conn, fs_id, split="val", table_name="ds_1_val")

        count = catalog.drop_by_feature_set_name("batting_features")
        assert count == 3

        assert conn.execute("SELECT COUNT(*) FROM dataset WHERE feature_set_id = ?", (fs_id,)).fetchone()[0] == 0

    def test_no_match_returns_zero(self, catalog: DatasetCatalogService) -> None:
        assert catalog.drop_by_feature_set_name("nonexistent") == 0

    def test_skips_missing_table(self, conn: sqlite3.Connection, catalog: DatasetCatalogService) -> None:
        """If the ds_N table was already manually dropped, don't error."""
        fs_id = _seed_feature_set(conn, "batting_features", "abc123")
        ds_id = _seed_dataset(conn, fs_id)
        table_name = f"ds_{ds_id}"

        # Manually drop the table first
        conn.execute(f"DROP TABLE [{table_name}]")

        count = catalog.drop_by_feature_set_name("batting_features")
        assert count == 1

    def test_does_not_orphan_feature_set_with_remaining_datasets(
        self, conn: sqlite3.Connection, catalog: DatasetCatalogService
    ) -> None:
        """If a feature_set has multiple versions, only drop the matching one."""
        fs1 = _seed_feature_set(conn, "batting_features", "v1")
        _seed_dataset(conn, fs1)
        fs2 = _seed_feature_set(conn, "batting_features", "v2")
        _seed_dataset(conn, fs2)

        catalog.drop_by_feature_set_name("batting_features")

        # Both versions should be cleaned up since they share the name
        assert conn.execute("SELECT COUNT(*) FROM feature_set WHERE name = ?", ("batting_features",)).fetchone()[0] == 0


class TestDropByNamePrefix:
    def test_drops_matching_prefix(self, conn: sqlite3.Connection, catalog: DatasetCatalogService) -> None:
        fs1 = _seed_feature_set(conn, "statcast_gbm_batting", "abc")
        _seed_dataset(conn, fs1)
        fs2 = _seed_feature_set(conn, "statcast_gbm_pitching", "def")
        _seed_dataset(conn, fs2)
        fs3 = _seed_feature_set(conn, "marcel_batting", "ghi")
        _seed_dataset(conn, fs3)

        count = catalog.drop_by_name_prefix("statcast_gbm_")
        assert count == 2

        # marcel should be untouched
        assert conn.execute("SELECT COUNT(*) FROM feature_set WHERE name = ?", ("marcel_batting",)).fetchone()[0] == 1

    def test_no_match_returns_zero(self, catalog: DatasetCatalogService) -> None:
        assert catalog.drop_by_name_prefix("nonexistent_") == 0


class TestDropAll:
    def test_drops_everything(self, conn: sqlite3.Connection, catalog: DatasetCatalogService) -> None:
        fs1 = _seed_feature_set(conn, "batting_features", "abc")
        ds1_id = _seed_dataset(conn, fs1)
        fs2 = _seed_feature_set(conn, "pitching_features", "def")
        ds2_id = _seed_dataset(conn, fs2)

        count = catalog.drop_all()
        assert count == 2

        assert not _table_exists(conn, f"ds_{ds1_id}")
        assert not _table_exists(conn, f"ds_{ds2_id}")
        assert conn.execute("SELECT COUNT(*) FROM dataset").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM feature_set").fetchone()[0] == 0

    def test_empty_returns_zero(self, catalog: DatasetCatalogService) -> None:
        assert catalog.drop_all() == 0


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None
