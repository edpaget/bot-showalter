"""Tests for minors/types.py"""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.minors.types import (
    MinorLeagueBatterSeasonStats,
    MinorLeagueLevel,
    MinorLeaguePitcherSeasonStats,
    MLEPrediction,
)


class TestMinorLeagueLevel:
    """Tests for MinorLeagueLevel enum."""

    def test_sport_id_values(self) -> None:
        assert MinorLeagueLevel.AAA.value == 11
        assert MinorLeagueLevel.AA.value == 12
        assert MinorLeagueLevel.HIGH_A.value == 13
        assert MinorLeagueLevel.SINGLE_A.value == 14
        assert MinorLeagueLevel.ROOKIE.value == 16

    def test_from_sport_id(self) -> None:
        assert MinorLeagueLevel.from_sport_id(11) == MinorLeagueLevel.AAA
        assert MinorLeagueLevel.from_sport_id(12) == MinorLeagueLevel.AA
        assert MinorLeagueLevel.from_sport_id(13) == MinorLeagueLevel.HIGH_A
        assert MinorLeagueLevel.from_sport_id(14) == MinorLeagueLevel.SINGLE_A
        assert MinorLeagueLevel.from_sport_id(16) == MinorLeagueLevel.ROOKIE

    def test_from_sport_id_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown sport_id"):
            MinorLeagueLevel.from_sport_id(99)

    def test_from_code(self) -> None:
        assert MinorLeagueLevel.from_code("AAA") == MinorLeagueLevel.AAA
        assert MinorLeagueLevel.from_code("AA") == MinorLeagueLevel.AA
        assert MinorLeagueLevel.from_code("A+") == MinorLeagueLevel.HIGH_A
        assert MinorLeagueLevel.from_code("HIGH-A") == MinorLeagueLevel.HIGH_A
        assert MinorLeagueLevel.from_code("A") == MinorLeagueLevel.SINGLE_A
        assert MinorLeagueLevel.from_code("SINGLE-A") == MinorLeagueLevel.SINGLE_A
        assert MinorLeagueLevel.from_code("Rk") == MinorLeagueLevel.ROOKIE
        assert MinorLeagueLevel.from_code("ROOKIE") == MinorLeagueLevel.ROOKIE

    def test_from_code_case_insensitive(self) -> None:
        assert MinorLeagueLevel.from_code("aaa") == MinorLeagueLevel.AAA
        assert MinorLeagueLevel.from_code("aa") == MinorLeagueLevel.AA
        assert MinorLeagueLevel.from_code("a+") == MinorLeagueLevel.HIGH_A

    def test_from_code_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown level code"):
            MinorLeagueLevel.from_code("INVALID")

    def test_display_name(self) -> None:
        assert MinorLeagueLevel.AAA.display_name == "AAA"
        assert MinorLeagueLevel.AA.display_name == "AA"
        assert MinorLeagueLevel.HIGH_A.display_name == "A+"
        assert MinorLeagueLevel.SINGLE_A.display_name == "A"
        assert MinorLeagueLevel.ROOKIE.display_name == "Rookie"


class TestMinorLeagueBatterSeasonStats:
    """Tests for MinorLeagueBatterSeasonStats dataclass."""

    def test_create_stats(self) -> None:
        stats = MinorLeagueBatterSeasonStats(
            player_id="12345",
            name="Test Player",
            season=2024,
            age=23,
            level=MinorLeagueLevel.AAA,
            team="Test Team",
            league="International League",
            pa=500,
            ab=450,
            h=130,
            singles=90,
            doubles=25,
            triples=5,
            hr=10,
            rbi=60,
            r=70,
            bb=40,
            so=100,
            hbp=5,
            sf=5,
            sb=15,
            cs=5,
            avg=0.289,
            obp=0.360,
            slg=0.440,
        )

        assert stats.player_id == "12345"
        assert stats.name == "Test Player"
        assert stats.season == 2024
        assert stats.age == 23
        assert stats.level == MinorLeagueLevel.AAA
        assert stats.pa == 500
        assert stats.hr == 10
        assert stats.sport_id == 11

    def test_sport_id_property(self) -> None:
        stats = MinorLeagueBatterSeasonStats(
            player_id="1",
            name="Player",
            season=2024,
            age=25,
            level=MinorLeagueLevel.AA,
            team="Team",
            league="Eastern League",
            pa=100,
            ab=90,
            h=25,
            singles=20,
            doubles=3,
            triples=1,
            hr=1,
            rbi=10,
            r=15,
            bb=8,
            so=20,
            hbp=2,
            sf=0,
            sb=5,
            cs=1,
            avg=0.278,
            obp=0.350,
            slg=0.389,
        )
        assert stats.sport_id == 12

    def test_frozen(self) -> None:
        stats = MinorLeagueBatterSeasonStats(
            player_id="1",
            name="Player",
            season=2024,
            age=25,
            level=MinorLeagueLevel.AAA,
            team="Team",
            league="League",
            pa=100,
            ab=90,
            h=25,
            singles=20,
            doubles=3,
            triples=1,
            hr=1,
            rbi=10,
            r=15,
            bb=8,
            so=20,
            hbp=2,
            sf=0,
            sb=5,
            cs=1,
            avg=0.278,
            obp=0.350,
            slg=0.389,
        )
        with pytest.raises(AttributeError):
            stats.pa = 200  # type: ignore[misc]


class TestMinorLeaguePitcherSeasonStats:
    """Tests for MinorLeaguePitcherSeasonStats dataclass."""

    def test_create_stats(self) -> None:
        stats = MinorLeaguePitcherSeasonStats(
            player_id="54321",
            name="Test Pitcher",
            season=2024,
            age=24,
            level=MinorLeagueLevel.AA,
            team="Test Team",
            league="Eastern League",
            g=25,
            gs=25,
            ip=140.0,
            w=10,
            losses=5,
            sv=0,
            h=120,
            r=55,
            er=50,
            hr=12,
            bb=40,
            so=150,
            hbp=5,
            era=3.21,
            whip=1.14,
        )

        assert stats.player_id == "54321"
        assert stats.name == "Test Pitcher"
        assert stats.ip == 140.0
        assert stats.so == 150
        assert stats.sport_id == 12

    def test_batters_faced_property(self) -> None:
        stats = MinorLeaguePitcherSeasonStats(
            player_id="1",
            name="Pitcher",
            season=2024,
            age=24,
            level=MinorLeagueLevel.AAA,
            team="Team",
            league="League",
            g=10,
            gs=10,
            ip=60.0,
            w=5,
            losses=3,
            sv=0,
            h=50,
            r=20,
            er=18,
            hr=5,
            bb=15,
            so=55,
            hbp=3,
            era=2.70,
            whip=1.08,
        )
        # BF ≈ IP * 3 + H + BB + HBP = 180 + 50 + 15 + 3 = 248
        assert stats.batters_faced == 248.0


class TestMLEPrediction:
    """Tests for MLEPrediction dataclass."""

    def test_create_prediction(self) -> None:
        pred = MLEPrediction(
            player_id="12345",
            source_season=2024,
            source_level=MinorLeagueLevel.AAA,
            source_pa=500,
            predicted_rates={"hr": 0.035, "so": 0.22, "bb": 0.10},
            confidence=0.85,
        )

        assert pred.player_id == "12345"
        assert pred.source_season == 2024
        assert pred.source_level == MinorLeagueLevel.AAA
        assert pred.source_pa == 500
        assert pred.predicted_rates["hr"] == 0.035
        assert pred.confidence == 0.85
