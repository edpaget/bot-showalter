from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.features.transforms.batted_ball import BATTED_BALL
from fantasy_baseball_manager.features.transforms.pitch_mix import PITCH_MIX
from fantasy_baseball_manager.features.types import (
    Feature,
    FeatureSet,
    Source,
    SpineFilter,
    TransformFeature,
)


def _noop_transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"out_a": 1.0}


NOOP_TRANSFORM = TransformFeature(
    name="noop",
    source=Source.BATTING,
    columns=("hr",),
    group_by=("player_id", "season"),
    transform=_noop_transform,
    outputs=("out_a",),
)


class TestMixedFeatureSet:
    """FeatureSet with both regular Feature and TransformFeature."""

    def test_materialize_has_all_columns(self, seeded_conn: sqlite3.Connection) -> None:
        regular = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
        fs = FeatureSet(
            name="test_mixed",
            features=(regular, NOOP_TRANSFORM),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        # Regular feature column
        assert "hr_1" in rows[0]
        # Transform output column
        assert "out_a" in rows[0]

    def test_transform_values_populated(self, seeded_conn: sqlite3.Connection) -> None:
        regular = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
        fs = FeatureSet(
            name="test_mixed",
            features=(regular, NOOP_TRANSFORM),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        for row in rows:
            assert row["out_a"] == pytest.approx(1.0)


class TestTransformOnly:
    """FeatureSet with only TransformFeature, no regular features."""

    def test_materialize_creates_table(self, seeded_conn: sqlite3.Connection) -> None:
        fs = FeatureSet(
            name="test_transform_only",
            features=(NOOP_TRANSFORM,),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        assert "player_id" in rows[0]
        assert "season" in rows[0]
        assert "out_a" in rows[0]
        for row in rows:
            assert row["out_a"] == pytest.approx(1.0)


class TestStatcastIntegration:
    """Integration tests with ATTACH for statcast DB."""

    def test_pitch_mix_materialize(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        fs = FeatureSet(
            name="test_statcast",
            features=(PITCH_MIX,),
            seasons=(2022, 2023),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        # 2 players x 2 seasons = 4 rows
        assert len(rows) == 4
        assert "ff_pct" in rows[0]
        assert "ff_velo" in rows[0]
        # All output columns present
        for output in PITCH_MIX.outputs:
            assert output in rows[0]

    def test_pitch_mix_values(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        fs = FeatureSet(
            name="test_statcast_values",
            features=(PITCH_MIX,),
            seasons=(2022,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}
        # Player 1 (id=1) in 2022: FF x3, SL x1, CH x1, CU x1 = 6 pitches
        p1 = by_player[1]
        assert p1["ff_pct"] == pytest.approx(3 / 6 * 100)
        assert p1["ff_velo"] == pytest.approx((95.0 + 97.0 + 96.0) / 3)
        assert p1["sl_pct"] == pytest.approx(1 / 6 * 100)
        assert p1["ch_pct"] == pytest.approx(1 / 6 * 100)
        assert p1["cu_pct"] == pytest.approx(1 / 6 * 100)

    def test_batted_ball_materialize(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        fs = FeatureSet(
            name="test_batted_ball",
            features=(BATTED_BALL,),
            seasons=(2022,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        for output in BATTED_BALL.outputs:
            assert output in rows[0]
        by_player = {r["player_id"]: r for r in rows}
        # Player 1 in 2022: batted balls at 100, 90, 105, 80 → 4 events
        p1 = by_player[1]
        assert p1["avg_exit_velo"] == pytest.approx((100 + 90 + 105 + 80) / 4)
        assert p1["max_exit_velo"] == pytest.approx(105.0)

    def test_mixed_regular_and_statcast(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        regular = Feature(name="hr_0", source=Source.BATTING, column="hr")
        fs = FeatureSet(
            name="test_mixed_statcast",
            features=(regular, PITCH_MIX),
            seasons=(2022,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        # Both regular and transform columns present
        assert "hr_0" in rows[0]
        assert "ff_pct" in rows[0]
