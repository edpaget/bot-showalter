from __future__ import annotations

import sqlite3

from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import (
    DatasetHandle,
    DatasetSplits,
    FeatureSet,
    Source,
    SourceRef,
)

batting = SourceRef(Source.BATTING)
player = SourceRef(Source.PLAYER)


def _simple_feature_set() -> FeatureSet:
    """Single direct feature: HR for season 2023, 2 players expected."""
    return FeatureSet(
        name="test_simple",
        features=(batting.col("hr").alias("hr"),),
        seasons=(2023,),
        source_filter="fangraphs",
    )


def _multi_feature_set() -> FeatureSet:
    """Mixed feature types for 2022-2023."""
    return FeatureSet(
        name="test_multi",
        features=(
            batting.col("hr").alias("hr_direct"),
            batting.col("h").per("ab").alias("batting_avg"),
            batting.col("hr").lag(1).rolling_mean(2).alias("hr_rolling"),
            player.age(),
        ),
        seasons=(2022, 2023),
        source_filter="fangraphs",
    )


class TestConstructor:
    def test_constructs_without_error(self, conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(conn)
        assert assembler is not None

    def test_is_dataset_assembler(self, conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(conn)
        assert isinstance(assembler, DatasetAssembler)


class TestMaterialize:
    def test_returns_dataset_handle(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        assert isinstance(handle, DatasetHandle)

    def test_table_name_starts_with_ds(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        assert handle.table_name.startswith("ds_")

    def test_row_count_matches(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        # 2 players x 1 season (2023)
        assert handle.row_count == 2

    def test_seasons_match(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        assert handle.seasons == (2023,)

    def test_ids_are_positive(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        assert handle.feature_set_id > 0
        assert handle.dataset_id > 0

    def test_feature_set_row_in_db(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        fs = _simple_feature_set()
        assembler.materialize(fs)
        row = seeded_conn.execute(
            "SELECT name, version, source_query FROM feature_set WHERE name = ?",
            (fs.name,),
        ).fetchone()
        assert row is not None
        assert row[0] == fs.name
        assert row[1] == fs.version
        assert row[2] is not None  # source_query stored

    def test_dataset_row_in_db(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        row = seeded_conn.execute(
            "SELECT feature_set_id, table_name, row_count FROM dataset WHERE id = ?",
            (handle.dataset_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == handle.feature_set_id
        assert row[1] == handle.table_name
        assert row[2] == handle.row_count

    def test_materialized_table_is_queryable(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        rows = seeded_conn.execute(f"SELECT * FROM {handle.table_name}").fetchall()
        assert len(rows) == handle.row_count


class TestMaterializeMultipleFeatures:
    def test_correct_column_count(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_feature_set())
        cursor = seeded_conn.execute(f"SELECT * FROM {handle.table_name} LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        # player_id, season, hr_direct, batting_avg, hr_rolling, age
        assert len(columns) == 6

    def test_row_count_multi_season(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_feature_set())
        # 2 players x 2 seasons (2022, 2023)
        assert handle.row_count == 4


class TestRead:
    def test_returns_list_of_dicts(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        rows = assembler.read(handle)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_length_matches_row_count(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        rows = assembler.read(handle)
        assert len(rows) == handle.row_count

    def test_keys_include_expected_columns(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        rows = assembler.read(handle)
        assert "player_id" in rows[0]
        assert "season" in rows[0]
        assert "hr" in rows[0]

    def test_values_match_expected_data(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_simple_feature_set())
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}
        # Trout 2023: hr=35, Betts 2023: hr=32
        assert by_player[1]["hr"] == 35
        assert by_player[2]["hr"] == 32


class TestGetOrMaterializeCacheHit:
    def test_second_call_returns_same_handle(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        fs = _simple_feature_set()
        h1 = assembler.get_or_materialize(fs)
        h2 = assembler.get_or_materialize(fs)
        assert h1.feature_set_id == h2.feature_set_id
        assert h1.table_name == h2.table_name

    def test_no_duplicate_feature_set_rows(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        fs = _simple_feature_set()
        assembler.get_or_materialize(fs)
        assembler.get_or_materialize(fs)
        count = seeded_conn.execute(
            "SELECT COUNT(*) FROM feature_set WHERE name = ? AND version = ?",
            (fs.name, fs.version),
        ).fetchone()[0]
        assert count == 1

    def test_no_duplicate_tables(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        fs = _simple_feature_set()
        assembler.get_or_materialize(fs)
        assembler.get_or_materialize(fs)
        tables = seeded_conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'ds_%'"
        ).fetchone()[0]
        # Only one ds_ table should exist
        assert tables == 1


class TestGetOrMaterializeCacheMiss:
    def test_different_features_produce_new_materialization(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        fs1 = _simple_feature_set()
        fs2 = FeatureSet(
            name="test_simple",
            features=(batting.col("bb").alias("bb"),),
            seasons=(2023,),
            source_filter="fangraphs",
        )
        h1 = assembler.get_or_materialize(fs1)
        h2 = assembler.get_or_materialize(fs2)
        assert h1.feature_set_id != h2.feature_set_id
        assert h1.table_name != h2.table_name

    def test_two_distinct_feature_set_rows(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        fs1 = _simple_feature_set()
        fs2 = FeatureSet(
            name="test_simple",
            features=(batting.col("bb").alias("bb"),),
            seasons=(2023,),
            source_filter="fangraphs",
        )
        assembler.get_or_materialize(fs1)
        assembler.get_or_materialize(fs2)
        count = seeded_conn.execute(
            "SELECT COUNT(*) FROM feature_set WHERE name = ?",
            (fs1.name,),
        ).fetchone()[0]
        assert count == 2


class TestGetOrMaterializeStaleCache:
    def test_rematerializes_when_table_dropped(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        fs = _simple_feature_set()
        h1 = assembler.get_or_materialize(fs)
        # Drop the materialized table
        seeded_conn.execute(f"DROP TABLE [{h1.table_name}]")
        seeded_conn.commit()
        h2 = assembler.get_or_materialize(fs)
        assert h2.dataset_id != h1.dataset_id
        assert h2.row_count == 2

    def test_new_table_is_queryable_after_rematerialization(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        fs = _simple_feature_set()
        h1 = assembler.get_or_materialize(fs)
        seeded_conn.execute(f"DROP TABLE [{h1.table_name}]")
        seeded_conn.commit()
        h2 = assembler.get_or_materialize(fs)
        rows = seeded_conn.execute(f"SELECT * FROM [{h2.table_name}]").fetchall()
        assert len(rows) == h2.row_count


def _multi_season_feature_set() -> FeatureSet:
    """HR feature for seasons 2020-2023 (2 players x 4 seasons = 8 rows)."""
    return FeatureSet(
        name="test_split",
        features=(batting.col("hr").alias("hr"),),
        seasons=(2020, 2021, 2022, 2023),
        source_filter="fangraphs",
    )


class TestSplitBasicPartitioning:
    def test_creates_train_and_validation(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        splits = assembler.split(handle, train=[2022], validation=[2023])
        assert isinstance(splits, DatasetSplits)
        assert splits.train is not None
        assert splits.validation is not None
        assert splits.holdout is None

    def test_train_row_count(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        splits = assembler.split(handle, train=[2022], validation=[2023])
        # 2 players x 1 season (2022)
        assert splits.train.row_count == 2

    def test_validation_row_count(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        splits = assembler.split(handle, train=[2022], validation=[2023])
        # 2 players x 1 season (2023)
        assert splits.validation is not None
        assert splits.validation.row_count == 2

    def test_split_tables_are_queryable(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        splits = assembler.split(handle, train=[2022], validation=[2023])
        train_rows = seeded_conn.execute(f"SELECT * FROM [{splits.train.table_name}]").fetchall()
        assert len(train_rows) == splits.train.row_count
        assert splits.validation is not None
        val_rows = seeded_conn.execute(f"SELECT * FROM [{splits.validation.table_name}]").fetchall()
        assert len(val_rows) == splits.validation.row_count

    def test_dataset_rows_have_correct_split_values(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        splits = assembler.split(handle, train=[2022], validation=[2023])
        train_row = seeded_conn.execute("SELECT split FROM dataset WHERE id = ?", (splits.train.dataset_id,)).fetchone()
        assert train_row[0] == "train"
        assert splits.validation is not None
        val_row = seeded_conn.execute(
            "SELECT split FROM dataset WHERE id = ?", (splits.validation.dataset_id,)
        ).fetchone()
        assert val_row[0] == "val"


class TestSplitAllThree:
    def test_all_three_splits_populated(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        splits = assembler.split(
            handle,
            train=range(2021, 2023),
            validation=[2023],
            holdout=[2020],
        )
        assert splits.train is not None
        assert splits.validation is not None
        assert splits.holdout is not None

    def test_row_counts_add_up(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        splits = assembler.split(
            handle,
            train=range(2021, 2023),
            validation=[2023],
            holdout=[2020],
        )
        # train: 2021, 2022 → 2 players x 2 seasons = 4
        assert splits.train.row_count == 4
        # validation: 2023 → 2 players x 1 season = 2
        assert splits.validation is not None
        assert splits.validation.row_count == 2
        # holdout: 2020 → 2 players x 1 season = 2
        assert splits.holdout is not None
        assert splits.holdout.row_count == 2


class TestSplitTrainOnly:
    def test_train_only(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        splits = assembler.split(handle, train=[2022, 2023])
        assert splits.train.row_count == 4
        assert splits.validation is None
        assert splits.holdout is None


class TestSplitIdempotent:
    def test_split_twice_does_not_raise(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(_multi_season_feature_set())
        assembler.split(handle, train=[2022], holdout=[2023])
        splits = assembler.split(handle, train=[2022], holdout=[2023])
        assert splits.train.row_count == 2
        assert splits.holdout is not None
        assert splits.holdout.row_count == 2


class TestIntegrationFullWorkflow:
    def test_end_to_end(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)

        # Define a realistic feature set with multiple feature types
        fs = FeatureSet(
            name="marcel_like",
            features=(
                batting.col("hr").alias("hr"),
                batting.col("h").per("ab").alias("avg"),
                batting.col("hr").lag(1).rolling_mean(2).alias("hr_rolling"),
                player.age(),
            ),
            seasons=(2021, 2022, 2023),
            source_filter="fangraphs",
        )

        # get_or_materialize → first call materializes
        handle = assembler.get_or_materialize(fs)
        assert handle.row_count == 6  # 2 players x 3 seasons

        # read → verify data
        rows = assembler.read(handle)
        assert len(rows) == 6
        assert all("player_id" in r for r in rows)
        assert all("hr" in r for r in rows)
        assert all("avg" in r for r in rows)
        assert all("age" in r for r in rows)

        # split → train/val/holdout
        splits = assembler.split(
            handle,
            train=[2021, 2022],
            validation=[2023],
        )
        assert splits.train.row_count == 4
        assert splits.validation is not None
        assert splits.validation.row_count == 2
        assert splits.holdout is None

        # read each split → verify per-split data
        train_rows = assembler.read(splits.train)
        assert len(train_rows) == 4
        train_seasons = {r["season"] for r in train_rows}
        assert train_seasons == {2021, 2022}

        val_rows = assembler.read(splits.validation)
        assert len(val_rows) == 2
        val_seasons = {r["season"] for r in val_rows}
        assert val_seasons == {2023}

        # Second get_or_materialize → cache hit
        handle2 = assembler.get_or_materialize(fs)
        assert handle2.feature_set_id == handle.feature_set_id
        assert handle2.table_name == handle.table_name
        assert handle2.dataset_id == handle.dataset_id
