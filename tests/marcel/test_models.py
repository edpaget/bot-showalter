from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)


class TestBattingSeasonStats:
    def test_construction(self) -> None:
        stats = BattingSeasonStats(
            player_id="abc123",
            name="Test Player",
            year=2024,
            age=28,
            pa=600,
            ab=540,
            h=160,
            singles=100,
            doubles=30,
            triples=5,
            hr=25,
            bb=50,
            so=120,
            hbp=5,
            sf=3,
            sh=2,
            sb=10,
            cs=3,
            r=80,
            rbi=90,
        )
        assert stats.player_id == "abc123"
        assert stats.pa == 600
        assert stats.hr == 25

    def test_frozen(self) -> None:
        stats = BattingSeasonStats(
            player_id="abc123",
            name="Test Player",
            year=2024,
            age=28,
            pa=600,
            ab=540,
            h=160,
            singles=100,
            doubles=30,
            triples=5,
            hr=25,
            bb=50,
            so=120,
            hbp=5,
            sf=3,
            sh=2,
            sb=10,
            cs=3,
            r=80,
            rbi=90,
        )
        with pytest.raises(FrozenInstanceError):
            stats.pa = 700  # type: ignore[misc]


class TestPitchingSeasonStats:
    def test_construction(self) -> None:
        stats = PitchingSeasonStats(
            player_id="xyz789",
            name="Test Pitcher",
            year=2024,
            age=30,
            ip=180.0,
            g=32,
            gs=32,
            er=70,
            h=150,
            bb=50,
            so=200,
            hr=20,
            hbp=5,
            w=0,
            sv=0,
            hld=0,
            bs=0,
        )
        assert stats.player_id == "xyz789"
        assert stats.ip == 180.0
        assert stats.gs == 32

    def test_frozen(self) -> None:
        stats = PitchingSeasonStats(
            player_id="xyz789",
            name="Test Pitcher",
            year=2024,
            age=30,
            ip=180.0,
            g=32,
            gs=32,
            er=70,
            h=150,
            bb=50,
            so=200,
            hr=20,
            hbp=5,
            w=0,
            sv=0,
            hld=0,
            bs=0,
        )
        with pytest.raises(FrozenInstanceError):
            stats.ip = 200.0  # type: ignore[misc]


class TestBattingProjection:
    def test_construction(self) -> None:
        proj = BattingProjection(
            player_id="abc123",
            name="Test Player",
            year=2025,
            age=29,
            pa=550.0,
            ab=495.0,
            h=140.0,
            singles=85.0,
            doubles=28.0,
            triples=4.0,
            hr=23.0,
            bb=45.0,
            so=110.0,
            hbp=4.5,
            sf=2.5,
            sh=1.5,
            sb=8.0,
            cs=2.5,
            r=70.0,
            rbi=85.0,
        )
        assert proj.year == 2025
        assert proj.pa == 550.0

    def test_frozen(self) -> None:
        proj = BattingProjection(
            player_id="abc123",
            name="Test Player",
            year=2025,
            age=29,
            pa=550.0,
            ab=495.0,
            h=140.0,
            singles=85.0,
            doubles=28.0,
            triples=4.0,
            hr=23.0,
            bb=45.0,
            so=110.0,
            hbp=4.5,
            sf=2.5,
            sh=1.5,
            sb=8.0,
            cs=2.5,
            r=70.0,
            rbi=85.0,
        )
        with pytest.raises(FrozenInstanceError):
            proj.pa = 600.0  # type: ignore[misc]


class TestPitchingProjection:
    def test_construction(self) -> None:
        proj = PitchingProjection(
            player_id="xyz789",
            name="Test Pitcher",
            year=2025,
            age=31,
            ip=170.0,
            g=30.0,
            gs=30.0,
            er=65.0,
            h=140.0,
            bb=45.0,
            so=190.0,
            hr=18.0,
            hbp=4.5,
            era=3.44,
            whip=1.088,
            w=0.0,
            nsvh=0.0,
        )
        assert proj.era == 3.44
        assert proj.whip == 1.088

    def test_frozen(self) -> None:
        proj = PitchingProjection(
            player_id="xyz789",
            name="Test Pitcher",
            year=2025,
            age=31,
            ip=170.0,
            g=30.0,
            gs=30.0,
            er=65.0,
            h=140.0,
            bb=45.0,
            so=190.0,
            hr=18.0,
            hbp=4.5,
            era=3.44,
            whip=1.088,
            w=0.0,
            nsvh=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            proj.era = 4.0  # type: ignore[misc]
