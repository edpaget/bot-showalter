from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.features.consensus_pt import (
    batting_consensus_features,
    build_consensus_features,
    make_consensus_transform,
    make_weighted_consensus_transform,
    pitching_consensus_features,
)
from fantasy_baseball_manager.features.types import FeatureSet, SpineFilter
from tests.features.conftest import (
    seed_batting_data,
    seed_pitching_data,
    seed_projection_data,
)

if TYPE_CHECKING:
    import sqlite3


class TestMakeConsensusTransform:
    """Unit tests for the consensus transform callable."""

    def _call(
        self,
        steamer_key: str,
        zips_key: str,
        output_key: str,
        steamer_val: float | None,
        zips_val: float | None,
    ) -> dict[str, Any]:
        transform = make_consensus_transform(steamer_key, zips_key, output_key)
        row: dict[str, Any] = {steamer_key: steamer_val, zips_key: zips_val}
        return transform([row])

    def test_both_present_returns_average(self) -> None:
        result = self._call("steamer_pa", "zips_pa", "consensus_pa", 620.0, 580.0)
        assert result["consensus_pa"] == pytest.approx(600.0)

    def test_only_steamer_returns_steamer(self) -> None:
        result = self._call("steamer_pa", "zips_pa", "consensus_pa", 620.0, None)
        assert result["consensus_pa"] == pytest.approx(620.0)

    def test_only_zips_returns_zips(self) -> None:
        result = self._call("steamer_pa", "zips_pa", "consensus_pa", None, 580.0)
        assert result["consensus_pa"] == pytest.approx(580.0)

    def test_neither_present_returns_nan(self) -> None:
        result = self._call("steamer_pa", "zips_pa", "consensus_pa", None, None)
        assert math.isnan(result["consensus_pa"])

    def test_steamer_nan_falls_back_to_zips(self) -> None:
        result = self._call("steamer_pa", "zips_pa", "consensus_pa", float("nan"), 580.0)
        assert result["consensus_pa"] == pytest.approx(580.0)

    def test_zips_nan_falls_back_to_steamer(self) -> None:
        result = self._call("steamer_pa", "zips_pa", "consensus_pa", 620.0, float("nan"))
        assert result["consensus_pa"] == pytest.approx(620.0)

    def test_ip_both_present_returns_average(self) -> None:
        result = self._call("steamer_ip", "zips_ip", "consensus_ip", 200.0, 190.0)
        assert result["consensus_ip"] == pytest.approx(195.0)

    def test_ip_only_steamer(self) -> None:
        result = self._call("steamer_ip", "zips_ip", "consensus_ip", 200.0, None)
        assert result["consensus_ip"] == pytest.approx(200.0)

    def test_ip_neither_present(self) -> None:
        result = self._call("steamer_ip", "zips_ip", "consensus_ip", None, None)
        assert math.isnan(result["consensus_ip"])


class TestMakeWeightedConsensusTransform:
    """Unit tests for the weighted consensus transform."""

    def _call(
        self,
        source_keys: list[tuple[str, float]],
        output_key: str,
        values: dict[str, float | None],
    ) -> dict[str, Any]:
        transform = make_weighted_consensus_transform(source_keys, output_key)
        return transform([values])

    def test_three_systems_equal_weight(self) -> None:
        result = self._call(
            [("a_pa", 1.0), ("b_pa", 1.0), ("c_pa", 1.0)],
            "consensus_pa",
            {"a_pa": 600.0, "b_pa": 500.0, "c_pa": 400.0},
        )
        assert result["consensus_pa"] == pytest.approx(500.0)

    def test_unequal_weights(self) -> None:
        result = self._call(
            [("a_pa", 2.0), ("b_pa", 1.0)],
            "consensus_pa",
            {"a_pa": 600.0, "b_pa": 300.0},
        )
        assert result["consensus_pa"] == pytest.approx(500.0)

    def test_missing_one_renormalizes(self) -> None:
        result = self._call(
            [("a_pa", 1.0), ("b_pa", 1.0), ("c_pa", 1.0)],
            "consensus_pa",
            {"a_pa": 600.0, "b_pa": None, "c_pa": 400.0},
        )
        assert result["consensus_pa"] == pytest.approx(500.0)

    def test_all_missing(self) -> None:
        result = self._call(
            [("a_pa", 1.0), ("b_pa", 1.0)],
            "consensus_pa",
            {"a_pa": None, "b_pa": None},
        )
        assert math.isnan(result["consensus_pa"])

    def test_single_system(self) -> None:
        result = self._call(
            [("a_pa", 1.0)],
            "consensus_pa",
            {"a_pa": 550.0},
        )
        assert result["consensus_pa"] == pytest.approx(550.0)

    def test_nan_values_treated_as_missing(self) -> None:
        result = self._call(
            [("a_pa", 1.0), ("b_pa", 1.0)],
            "consensus_pa",
            {"a_pa": float("nan"), "b_pa": 500.0},
        )
        assert result["consensus_pa"] == pytest.approx(500.0)


class TestBuildConsensusFeatures:
    """Tests for the build_consensus_features factory."""

    def test_default_produces_same_as_current(self) -> None:
        proj_features, consensus = build_consensus_features("pa")
        assert len(proj_features) == 2
        assert proj_features[0].name == "steamer_pa"
        assert proj_features[1].name == "zips_pa"
        assert consensus.name == "consensus_pa"
        assert "consensus_pa" in consensus.outputs

    def test_three_systems(self) -> None:
        systems = (("steamer", 1.0), ("zips", 1.0), ("atc", 1.0))
        proj_features, consensus = build_consensus_features("pa", systems=systems)
        assert len(proj_features) == 3
        names = [f.name for f in proj_features]
        assert names == ["steamer_pa", "zips_pa", "atc_pa"]
        assert consensus.name == "consensus_pa"
        assert set(consensus.inputs) == {"steamer_pa", "zips_pa", "atc_pa"}

    def test_ip_stat(self) -> None:
        proj_features, consensus = build_consensus_features("ip")
        assert proj_features[0].name == "steamer_ip"
        assert proj_features[1].name == "zips_ip"
        assert consensus.name == "consensus_ip"

    def test_single_system(self) -> None:
        systems = (("atc", 1.0),)
        proj_features, consensus = build_consensus_features("pa", systems=systems)
        assert len(proj_features) == 1
        assert proj_features[0].name == "atc_pa"


class TestBattingConsensusAssembler:
    """Integration: materialize consensus_pa from Steamer+ZiPS projection data."""

    def test_consensus_pa_averages_two_systems(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)
        seed_projection_data(conn)

        fs = FeatureSet(
            name="test_consensus_pa",
            features=tuple(batting_consensus_features()),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}

        # Player 1: steamer_pa=620, zips_pa=580 → consensus = 600
        assert by_player[1]["steamer_pa"] == pytest.approx(620.0)
        assert by_player[1]["zips_pa"] == pytest.approx(580.0)
        assert by_player[1]["consensus_pa"] == pytest.approx(600.0)

        # Player 2: steamer_pa=610, zips_pa=570 → consensus = 590
        assert by_player[2]["steamer_pa"] == pytest.approx(610.0)
        assert by_player[2]["zips_pa"] == pytest.approx(570.0)
        assert by_player[2]["consensus_pa"] == pytest.approx(590.0)

    def test_consensus_pa_fallback_single_system(self, conn: sqlite3.Connection) -> None:
        """When only one system covers a player, consensus falls back to that value."""
        seed_batting_data(conn)
        # Insert Steamer for both players but ZiPS only for player 1
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, pa)"
            " VALUES (1, 2023, 'steamer', '2023.1', 'batter', 620)"
        )
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, pa)"
            " VALUES (1, 2023, 'zips', '2023.1', 'batter', 580)"
        )
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, pa)"
            " VALUES (2, 2023, 'steamer', '2023.1', 'batter', 610)"
        )
        conn.commit()

        fs = FeatureSet(
            name="test_consensus_pa_fallback",
            features=tuple(batting_consensus_features()),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="batter"),
        )
        assembler = SqliteDatasetAssembler(conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}

        # Player 1: both systems → average
        assert by_player[1]["consensus_pa"] == pytest.approx(600.0)
        # Player 2: only Steamer → falls back to 610
        assert by_player[2]["consensus_pa"] == pytest.approx(610.0)


class TestPitchingConsensusAssembler:
    """Integration: materialize consensus_ip from Steamer+ZiPS pitcher projections."""

    def test_consensus_ip_averages_two_systems(self, conn: sqlite3.Connection) -> None:
        seed_batting_data(conn)
        seed_pitching_data(conn)
        # Insert pitcher projections with IP for players 3 and 4
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, ip)"
            " VALUES (3, 2023, 'steamer', '2023.1', 'pitcher', 200.0)"
        )
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, ip)"
            " VALUES (3, 2023, 'zips', '2023.1', 'pitcher', 190.0)"
        )
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, ip)"
            " VALUES (4, 2023, 'steamer', '2023.1', 'pitcher', 170.0)"
        )
        conn.execute(
            "INSERT INTO projection (player_id, season, system, version, player_type, ip)"
            " VALUES (4, 2023, 'zips', '2023.1', 'pitcher', 160.0)"
        )
        conn.commit()

        fs = FeatureSet(
            name="test_consensus_ip",
            features=tuple(pitching_consensus_features()),
            seasons=(2023,),
            source_filter="fangraphs",
            spine_filter=SpineFilter(player_type="pitcher"),
        )
        assembler = SqliteDatasetAssembler(conn)
        handle = assembler.materialize(fs)
        rows = assembler.read(handle)
        by_player = {r["player_id"]: r for r in rows}

        # Player 3: steamer_ip=200, zips_ip=190 → consensus = 195
        assert by_player[3]["consensus_ip"] == pytest.approx(195.0)
        # Player 4: steamer_ip=170, zips_ip=160 → consensus = 165
        assert by_player[4]["consensus_ip"] == pytest.approx(165.0)
