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
    build_batting_weighted_rates,
)


class TestWeightedRatesMaterialization:
    """End-to-end test: materialize a feature set with weighted-rates derived transform."""

    def test_columns_exist(self, seeded_conn: sqlite3.Connection) -> None:
        categories = ("hr", "h")
        weights = (5.0, 4.0, 3.0)
        base_features = build_batting_features(categories, lags=3)
        derived = build_batting_weighted_rates(categories, weights)
        fs = FeatureSet(
            name="test_wavg",
            features=(*base_features, derived),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        assert len(rows) == 2
        assert "hr_wavg" in rows[0]
        assert "h_wavg" in rows[0]
        assert "weighted_pt" in rows[0]

    def test_values_correct(self, seeded_conn: sqlite3.Connection) -> None:
        categories = ("hr",)
        weights = (5.0, 4.0, 3.0)
        base_features = build_batting_features(categories, lags=3)
        derived = build_batting_weighted_rates(categories, weights)
        fs = FeatureSet(
            name="test_wavg_values",
            features=(*base_features, derived),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}

        # Player 1, season 2023:
        #   hr_1=40 (2022, lag=1), hr_2=30 (2021, lag=2), hr_3=17 (2020, lag=3)
        #   pa_1=550, pa_2=500, pa_3=250
        #   weighted hr = 40*5 + 30*4 + 17*3 = 200+120+51 = 371
        #   weighted pa = 550*5 + 500*4 + 250*3 = 2750+2000+750 = 5500
        #   hr_wavg = 371/5500
        p1 = by_player[1]
        assert p1["hr_wavg"] == pytest.approx(371.0 / 5500.0)
        assert p1["weighted_pt"] == pytest.approx(5500.0)

        # Player 2, season 2023:
        #   hr_1=28 (2022, lag=1), hr_2=22 (2021, lag=2), hr_3=10 (2020, lag=3)
        #   pa_1=520, pa_2=480, pa_3=200
        #   weighted hr = 28*5 + 22*4 + 10*3 = 140+88+30 = 258
        #   weighted pa = 520*5 + 480*4 + 200*3 = 2600+1920+600 = 5120
        p2 = by_player[2]
        assert p2["hr_wavg"] == pytest.approx(258.0 / 5120.0)
        assert p2["weighted_pt"] == pytest.approx(5120.0)

    def test_multiple_categories(self, seeded_conn: sqlite3.Connection) -> None:
        categories = ("hr", "h", "bb")
        weights = (5.0, 4.0, 3.0)
        base_features = build_batting_features(categories, lags=3)
        derived = build_batting_weighted_rates(categories, weights)
        fs = FeatureSet(
            name="test_wavg_multi",
            features=(*base_features, derived),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(seeded_conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        for row in rows:
            assert row["hr_wavg"] is not None
            assert row["h_wavg"] is not None
            assert row["bb_wavg"] is not None
            assert row["weighted_pt"] > 0
