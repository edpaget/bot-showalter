from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.features.types import (
    DerivedTransformFeature,
    Feature,
    FeatureSet,
    Source,
    SpineFilter,
    TransformFeature,
)


def _sum_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum hr_1 and pa_1 for the group, returning a single derived value."""
    total = sum(r.get("hr_1", 0) or 0 for r in rows) + sum(r.get("pa_1", 0) or 0 for r in rows)
    return {"derived_total": float(total)}


def _noop_source_transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Source transform that produces a fixed value from raw batting data."""
    return {"src_out": 42.0}


def _derived_from_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derived transform that reads the source transform output."""
    val = rows[0].get("src_out") if rows else None
    return {"derived_from_src": float(val) if val is not None else 0.0}


def _season_avg_hr(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute season-level average HR across all players in the group."""
    hr_values = [r.get("hr_0", 0) or 0 for r in rows]
    avg = sum(hr_values) / len(hr_values) if hr_values else 0.0
    return {"season_avg_hr": float(avg)}


SUM_DERIVED = DerivedTransformFeature(
    name="sum_features",
    inputs=("hr_1", "pa_1"),
    group_by=("player_id", "season"),
    transform=_sum_inputs,
    outputs=("derived_total",),
)


class TestDerivedTransformMixed:
    """FeatureSet with regular Features + DerivedTransformFeature."""

    def test_materialize_has_all_columns(self, seeded_conn: sqlite3.Connection) -> None:
        hr_1 = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
        pa_1 = Feature(name="pa_1", source=Source.BATTING, column="pa", lag=1)
        fs = FeatureSet(
            name="test_derived_mixed",
            features=(hr_1, pa_1, SUM_DERIVED),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        assert "hr_1" in rows[0]
        assert "pa_1" in rows[0]
        assert "derived_total" in rows[0]

    def test_derived_transform_values_correct(self, seeded_conn: sqlite3.Connection) -> None:
        hr_1 = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
        pa_1 = Feature(name="pa_1", source=Source.BATTING, column="pa", lag=1)
        fs = FeatureSet(
            name="test_derived_values",
            features=(hr_1, pa_1, SUM_DERIVED),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}
        # Player 1 in 2023: hr_1=lag1 hr=40 (2022), pa_1=lag1 pa=550 (2022)
        # derived_total = 40 + 550 = 590
        assert by_player[1]["derived_total"] == pytest.approx(590.0)
        # Player 2 in 2023: hr_1=lag1 hr=28 (2022), pa_1=lag1 pa=520 (2022)
        # derived_total = 28 + 520 = 548
        assert by_player[2]["derived_total"] == pytest.approx(548.0)


class TestDerivedTransformOnly:
    """FeatureSet with only DerivedTransformFeature, no regular features."""

    def test_materialize_creates_table(self, seeded_conn: sqlite3.Connection) -> None:
        # A derived transform that reads from spine columns (player_id, season)
        def _const_transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return {"const_out": 99.0}

        dtf = DerivedTransformFeature(
            name="const_derived",
            inputs=("player_id",),
            group_by=("player_id", "season"),
            transform=_const_transform,
            outputs=("const_out",),
        )
        fs = FeatureSet(
            name="test_derived_only",
            features=(dtf,),
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
        assert "const_out" in rows[0]
        for row in rows:
            assert row["const_out"] == pytest.approx(99.0)


class TestDerivedSeesSourceTransformOutputs:
    """Ordering guarantee: derived transforms see source transform outputs."""

    def test_derived_receives_source_transform_values(self, seeded_conn: sqlite3.Connection) -> None:
        src_tf = TransformFeature(
            name="src_transform",
            source=Source.BATTING,
            columns=("hr",),
            group_by=("player_id", "season"),
            transform=_noop_source_transform,
            outputs=("src_out",),
        )
        derived_tf = DerivedTransformFeature(
            name="derived_from_source",
            inputs=("src_out",),
            group_by=("player_id", "season"),
            transform=_derived_from_source,
            outputs=("derived_from_src",),
        )
        fs = FeatureSet(
            name="test_ordering",
            features=(src_tf, derived_tf),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        for row in rows:
            # Source transform produces 42.0, derived reads it
            assert row["src_out"] == pytest.approx(42.0)
            assert row["derived_from_src"] == pytest.approx(42.0)


class TestDerivedGroupBySeason:
    """Cross-player aggregation with group_by=("season",)."""

    def test_broadcast_season_average(self, seeded_conn: sqlite3.Connection) -> None:
        hr_0 = Feature(name="hr_0", source=Source.BATTING, column="hr")
        derived_season = DerivedTransformFeature(
            name="season_avg",
            inputs=("hr_0",),
            group_by=("season",),
            transform=_season_avg_hr,
            outputs=("season_avg_hr",),
        )
        fs = FeatureSet(
            name="test_season_group",
            features=(hr_0, derived_season),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        # Player 1 hr=35, Player 2 hr=32 in 2023 → avg = (35+32)/2 = 33.5
        for row in rows:
            assert row["season_avg_hr"] == pytest.approx(33.5)
