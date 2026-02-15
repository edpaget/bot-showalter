from __future__ import annotations

import sqlite3

import pytest

from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.features.types import (
    FeatureSet,
    SpineFilter,
)
from fantasy_baseball_manager.models.marcel.features import (
    build_batting_features,
    build_batting_league_averages,
)


class TestLeagueAveragesMaterialization:
    """End-to-end: materialize with league-averages derived transform."""

    def test_columns_exist(self, seeded_conn: sqlite3.Connection) -> None:
        categories = ("hr", "h")
        base_features = build_batting_features(categories, lags=3)
        league = build_batting_league_averages(categories)
        fs = FeatureSet(
            name="test_league",
            features=(*base_features, league),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        assert "league_hr_rate" in rows[0]
        assert "league_h_rate" in rows[0]

    def test_all_rows_same_season_get_same_league_rate(self, seeded_conn: sqlite3.Connection) -> None:
        categories = ("hr",)
        base_features = build_batting_features(categories, lags=3)
        league = build_batting_league_averages(categories)
        fs = FeatureSet(
            name="test_league_broadcast",
            features=(*base_features, league),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        # Both rows are season 2023 — they must have the same league rate
        assert rows[0]["league_hr_rate"] == rows[1]["league_hr_rate"]

    def test_league_rate_values_correct(self, seeded_conn: sqlite3.Connection) -> None:
        categories = ("hr",)
        base_features = build_batting_features(categories, lags=3)
        league = build_batting_league_averages(categories)
        fs = FeatureSet(
            name="test_league_values",
            features=(*base_features, league),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        # League rate uses lag-1 values (2022 stats):
        #   Player 1: hr_1=40, pa_1=550
        #   Player 2: hr_1=28, pa_1=520
        #   league_hr_rate = (40+28) / (550+520) = 68/1070
        for row in rows:
            assert row["league_hr_rate"] == pytest.approx(68.0 / 1070.0)

    def test_multi_season_broadcast(self, seeded_conn: sqlite3.Connection) -> None:
        """Each season gets its own league average, broadcast to all rows in that season."""
        categories = ("hr",)
        base_features = build_batting_features(categories, lags=3)
        league = build_batting_league_averages(categories)
        fs = FeatureSet(
            name="test_league_multi_season",
            features=(*base_features, league),
            seasons=(2022, 2023),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)

        by_season: dict[int, list[dict[str, object]]] = {}
        for row in rows:
            by_season.setdefault(row["season"], []).append(row)

        # Within each season, all rows have the same league rate
        for season_rows in by_season.values():
            rates = [r["league_hr_rate"] for r in season_rows]
            assert all(r == rates[0] for r in rates)

        # Different seasons should have different league rates
        # (because they use different lag-1 stats)
        rate_2022 = by_season[2022][0]["league_hr_rate"]
        rate_2023 = by_season[2023][0]["league_hr_rate"]
        assert rate_2022 != rate_2023

        # 2022 uses lag-1 = 2021 stats:
        #   Player 1: hr_1=30, pa_1=500
        #   Player 2: hr_1=22, pa_1=480
        #   league_hr_rate = 52/980
        assert rate_2022 == pytest.approx(52.0 / 980.0)

        # 2023 uses lag-1 = 2022 stats:
        #   Player 1: hr_1=40, pa_1=550
        #   Player 2: hr_1=28, pa_1=520
        #   league_hr_rate = 68/1070
        assert rate_2023 == pytest.approx(68.0 / 1070.0)
