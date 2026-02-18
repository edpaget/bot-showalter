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


class TestLaggedStatcastTransform:
    """Test lagged TransformFeature joins statcast data from prior season."""

    def test_lag_1_joins_prior_season(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """With lag=1, a 2023 row should get 2022 statcast data."""
        lagged_batted_ball = BATTED_BALL.with_lag(1)
        fs = FeatureSet(
            name="test_lagged_statcast",
            features=(lagged_batted_ball,),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        # 2 players in season 2023, should get 2022 statcast data
        assert len(rows) == 2
        by_player = {r["player_id"]: r for r in rows}
        # Player 1 in 2023: should have 2022 batted ball data
        # Player 1 2022 batted balls: exit velos 100, 90, 105, 80 → avg 93.75
        p1 = by_player[1]
        assert p1["avg_exit_velo"] == pytest.approx((100 + 90 + 105 + 80) / 4)

    def test_lag_0_joins_same_season(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """With lag=0 (default), a 2022 row should get 2022 statcast data."""
        fs = FeatureSet(
            name="test_unlagged_statcast",
            features=(BATTED_BALL,),
            seasons=(2022,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}
        p1 = by_player[1]
        assert p1["avg_exit_velo"] == pytest.approx((100 + 90 + 105 + 80) / 4)


class TestAvgLagStatcastTransform:
    """Test with_avg_lag pools raw data from multiple lagged seasons."""

    def test_avg_lag_pools_two_seasons(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """with_avg_lag(1, 2) for season 2023 pools 2022 + 2021 statcast data."""
        avg_batted_ball = BATTED_BALL.with_avg_lag(1, 2)
        fs = FeatureSet(
            name="test_avg_lag",
            features=(avg_batted_ball,),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        by_player = {r["player_id"]: r for r in rows}
        # Player 1: 2021 batted balls [110, 70], 2022 batted balls [100, 90, 105, 80]
        # Pooled: [110, 70, 100, 90, 105, 80] → avg = 555/6 = 92.5
        p1 = by_player[1]
        assert p1["avg_exit_velo"] == pytest.approx((110 + 70 + 100 + 90 + 105 + 80) / 6)

    def test_avg_lag_single_year_fallback(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """When only one lag year has data, transform still produces valid output."""
        # Use with_avg_lag(1, 3) — lag-3 from season 2023 = 2020, no data exists
        avg_batted_ball = BATTED_BALL.with_avg_lag(1, 3)
        fs = FeatureSet(
            name="test_avg_lag_fallback",
            features=(avg_batted_ball,),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        by_player = {r["player_id"]: r for r in rows}
        # Player 1: only 2022 data (lag-1), lag-3 = 2020 has no data
        # 2022 batted balls: [100, 90, 105, 80] → avg = 93.75
        p1 = by_player[1]
        assert p1["avg_exit_velo"] == pytest.approx((100 + 90 + 105 + 80) / 4)

    def test_avg_lag_barrel_pct_is_pooled(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """barrel_pct is computed over pooled BIP, not averaged across seasons."""
        avg_batted_ball = BATTED_BALL.with_avg_lag(1, 2)
        fs = FeatureSet(
            name="test_avg_lag_barrel",
            features=(avg_batted_ball,),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}
        # Player 1: 2021 barrels [1, 0], 2022 barrels [1, 0, 1, 0]
        # Pooled: 3 barrels out of 6 BIP = 50.0%
        # (Not avg of 2021=50% and 2022=50% — happens to match here, but
        #  the mechanism is raw pooling)
        p1 = by_player[1]
        assert p1["barrel_pct"] == pytest.approx(3 / 6 * 100.0)

    def test_avg_lag_same_columns_as_single_lag(
        self,
        seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """Output column names identical to with_lag(1)."""
        single = BATTED_BALL.with_lag(1)
        avg = BATTED_BALL.with_avg_lag(1, 2)
        fs_single = FeatureSet(
            name="test_single",
            features=(single,),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        fs_avg = FeatureSet(
            name="test_avg",
            features=(avg,),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn, statcast_path=statcast_db_path)
        single_rows = assembler.read(assembler.materialize(fs_single))
        avg_rows = assembler.read(assembler.materialize(fs_avg))
        assert set(single_rows[0].keys()) == set(avg_rows[0].keys())


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

    def test_pitcher_pitch_mix_materialize(
        self,
        pitcher_seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """Pitcher feature set joins statcast on pitcher_id, returning rows."""
        fs = FeatureSet(
            name="test_pitcher_statcast",
            features=(PITCH_MIX,),
            seasons=(2022, 2023),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="pitcher"),
        )
        assembler = SqliteDatasetAssembler(pitcher_seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        # 2 pitchers x 2 seasons = 4 rows
        assert len(rows) == 4
        for output in PITCH_MIX.outputs:
            assert output in rows[0]
        # At least one pitcher should have non-zero ff_pct (proves pitcher_id join worked)
        assert any(r["ff_pct"] > 0 for r in rows)

    def test_pitcher_pitch_mix_values(
        self,
        pitcher_seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """Verify specific pitch mix values for a known pitcher."""
        fs = FeatureSet(
            name="test_pitcher_values",
            features=(PITCH_MIX,),
            seasons=(2022,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="pitcher"),
        )
        assembler = SqliteDatasetAssembler(pitcher_seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}
        # Pitcher player_id=3 (mlbam_id=100001) in 2022: pitched in game 1001
        # 3 pitches: FF (95.0), FF (97.0), SL (85.0)
        p3 = by_player[3]
        assert p3["ff_pct"] == pytest.approx(2 / 3 * 100)
        assert p3["ff_velo"] == pytest.approx((95.0 + 97.0) / 2)
        assert p3["sl_pct"] == pytest.approx(1 / 3 * 100)

    def test_batter_join_unaffected(
        self,
        pitcher_seeded_conn: sqlite3.Connection,
        statcast_db_path: Path,
    ) -> None:
        """Batter feature sets still join on batter_id after the pitcher fix."""
        fs = FeatureSet(
            name="test_batter_regression",
            features=(PITCH_MIX,),
            seasons=(2022,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(pitcher_seeded_conn, statcast_path=statcast_db_path)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}
        # Player 1 (batter, mlbam_id=545361) should still get batter-side statcast data
        p1 = by_player[1]
        assert p1["ff_pct"] == pytest.approx(3 / 6 * 100)
