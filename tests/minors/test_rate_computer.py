"""Tests for minors/rate_computer.py - MLE rate computer pipeline stage."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest

from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
from fantasy_baseball_manager.minors.model import MLEGradientBoostingModel, MLEStatModel
from fantasy_baseball_manager.minors.training_data import AggregatedMiLBStats
from fantasy_baseball_manager.minors.types import (
    MinorLeagueBatterSeasonStats,
    MinorLeagueLevel,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates


def _make_player(
    player_id: str = "123456",
    name: str = "Test Batter",
    year: int = 2024,
    age: int = 24,
    pa: int = 150,  # Low PA - needs MLE
    ab: int = 135,
    h: int = 35,
    singles: int = 20,
    doubles: int = 8,
    triples: int = 2,
    hr: int = 5,
    bb: int = 12,
    so: int = 40,
    hbp: int = 2,
    sf: int = 1,
    sh: int = 0,
    sb: int = 3,
    cs: int = 1,
    r: int = 20,
    rbi: int = 18,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        sh=sh,
        sb=sb,
        cs=cs,
        r=r,
        rbi=rbi,
    )


def _make_league(year: int = 2024) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        pa=6000,
        ab=5400,
        h=1500,
        singles=900,
        doubles=300,
        triples=30,
        hr=200,
        bb=500,
        so=1400,
        hbp=50,
        sf=30,
        sh=20,
        sb=100,
        cs=30,
        r=800,
        rbi=750,
    )


def _make_pitcher(
    player_id: str = "sp1",
    name: str = "Test Pitcher",
    year: int = 2024,
    age: int = 28,
    ip: float = 180.0,
    g: int = 32,
    gs: int = 32,
    er: int = 70,
    h: int = 150,
    bb: int = 50,
    so: int = 200,
    hr: int = 20,
    hbp: int = 5,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        ip=ip,
        g=g,
        gs=gs,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=hbp,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


def _make_league_pitching(year: int = 2024) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        ip=1450.0,
        g=500,
        gs=162,
        er=650,
        h=1350,
        bb=500,
        so=1400,
        hr=180,
        hbp=60,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


def _make_milb_batter(
    player_id: str = "123456",
    name: str = "Test Batter",
    season: int = 2023,
    age: int = 23,
    level: MinorLeagueLevel = MinorLeagueLevel.AAA,
    pa: int = 450,
) -> MinorLeagueBatterSeasonStats:
    """Create MiLB batter stats for testing."""
    ab = int(pa * 0.9)
    h = int(ab * 0.270)
    hr = int(pa * 0.035)
    bb = int(pa * 0.08)
    so = int(pa * 0.22)
    doubles = int(pa * 0.045)
    triples = int(pa * 0.005)
    singles = h - doubles - triples - hr
    return MinorLeagueBatterSeasonStats(
        player_id=player_id,
        name=name,
        season=season,
        age=age,
        level=level,
        team="AAA Team",
        league="IL",
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        rbi=int(pa * 0.12),
        r=int(pa * 0.13),
        bb=bb,
        so=so,
        hbp=int(pa * 0.01),
        sf=int(pa * 0.005),
        sb=int(pa * 0.02),
        cs=int(pa * 0.005),
        avg=0.270,
        obp=0.340,
        slg=0.450,
    )


class FakeMLBDataSource:
    """Fake MLB data source for testing."""

    def __init__(
        self,
        player_batting: dict[int, list[BattingSeasonStats]] | None = None,
        team_batting_data: dict[int, list[BattingSeasonStats]] | None = None,
        player_pitching: dict[int, list[PitchingSeasonStats]] | None = None,
        team_pitching_data: dict[int, list[PitchingSeasonStats]] | None = None,
    ) -> None:
        self._player_batting = player_batting or {}
        self._team_batting_data = team_batting_data or {}
        self._player_pitching = player_pitching or {}
        self._team_pitching_data = team_pitching_data or {}

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._player_batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._player_pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._team_batting_data.get(year, [])

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_pitching_data.get(year, [])


@dataclass
class FakeMinorLeagueDataSource:
    """Fake MiLB data source for testing."""

    batting_data: dict[int, list[MinorLeagueBatterSeasonStats]] = field(default_factory=dict)

    def batting_stats(self, year: int, level: MinorLeagueLevel) -> list[MinorLeagueBatterSeasonStats]:
        all_stats = self.batting_data.get(year, [])
        return [s for s in all_stats if s.level == level]

    def batting_stats_all_levels(self, year: int) -> list[MinorLeagueBatterSeasonStats]:
        return self.batting_data.get(year, [])


def _create_fitted_mle_model(feature_names: list[str]) -> MLEGradientBoostingModel:
    """Create a fitted MLE model for testing."""
    np.random.seed(42)
    n_features = len(feature_names)
    X = np.random.randn(50, n_features).astype(np.float32)

    model_set = MLEGradientBoostingModel(
        player_type="batter",
        feature_names=feature_names,
        training_years=(2022, 2023),
    )

    # Train models for each stat
    for stat in ["hr", "so", "bb", "singles", "doubles", "triples", "sb"]:
        y = np.random.randn(50) * 0.01 + 0.05
        model = MLEStatModel(stat_name=stat)
        model.fit(X, y, feature_names)
        model_set.add_model(model)

    return model_set


class TestMLERateComputerConfig:
    """Tests for MLERateComputerConfig."""

    def test_default_config(self) -> None:
        """Config should have sensible defaults."""
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputerConfig

        config = MLERateComputerConfig()

        assert config.model_name == "default"
        assert config.min_milb_pa == 200
        assert config.mlb_pa_threshold == 200

    def test_custom_config(self) -> None:
        """Config should accept custom values."""
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputerConfig

        config = MLERateComputerConfig(
            model_name="custom",
            min_milb_pa=300,
            mlb_pa_threshold=150,
        )

        assert config.model_name == "custom"
        assert config.min_milb_pa == 300
        assert config.mlb_pa_threshold == 150


class TestMLERateComputer:
    """Tests for MLERateComputer."""

    def test_init(self) -> None:
        """MLERateComputer should initialize correctly."""
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputer

        milb_source = MagicMock()
        model_store = MagicMock()
        model_store.exists.return_value = False

        computer = MLERateComputer(
            milb_source=milb_source,
            model_store=model_store,
        )

        assert computer.config.model_name == "default"
        assert computer.config.min_milb_pa == 200

    def test_custom_config(self) -> None:
        """MLERateComputer should accept custom config."""
        from fantasy_baseball_manager.minors.rate_computer import (
            MLERateComputer,
            MLERateComputerConfig,
        )

        config = MLERateComputerConfig(model_name="custom", mlb_pa_threshold=150)

        computer = MLERateComputer(
            milb_source=MagicMock(),
            model_store=MagicMock(),
            config=config,
        )

        assert computer.config.model_name == "custom"
        assert computer.config.mlb_pa_threshold == 150

    def test_falls_back_to_marcel_when_no_model(self) -> None:
        """When MLE model not found, should return Marcel rates."""
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputer

        mock_store = MagicMock()
        mock_store.exists.return_value = False

        computer = MLERateComputer(
            milb_source=MagicMock(),
            model_store=mock_store,
        )

        computer._ensure_model_loaded()
        mock_store.exists.assert_called()
        assert computer._batter_model is None

    def test_compute_batting_rates_uses_mle_for_low_mlb_pa(self) -> None:
        """Players with low MLB PA should get MLE predictions blended in."""
        from fantasy_baseball_manager.minors.features import MLEBatterFeatureExtractor
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputer

        extractor = MLEBatterFeatureExtractor(min_pa=50)
        feature_names = extractor.feature_names()
        mle_model = _create_fitted_mle_model(feature_names)

        # Player with low MLB PA (100) in prior year
        mlb_player = _make_player(
            player_id="123456",
            year=2024,
            age=24,
            pa=100,  # Below 200 threshold
        )
        league = _make_league(year=2024)

        mlb_source = FakeMLBDataSource(
            player_batting={
                2024: [mlb_player],
                2023: [],
                2022: [],
            },
            team_batting_data={
                2024: [league],
                2023: [_make_league(2023)],
                2022: [_make_league(2022)],
            },
        )

        # MiLB stats from prior year (2024 for 2025 projections)
        milb_player = _make_milb_batter(
            player_id="123456",
            season=2024,
            age=24,
            pa=450,
        )
        milb_source = FakeMinorLeagueDataSource(batting_data={2024: [milb_player]})

        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store.load.return_value = mle_model

        computer = MLERateComputer(
            milb_source=milb_source,
            model_store=mock_store,
        )

        rates = computer.compute_batting_rates(mlb_source, 2025, 3)

        assert len(rates) == 1
        assert rates[0].player_id == "123456"
        assert rates[0].year == 2025
        # MLE metadata should be present
        assert rates[0].metadata.get("mle_applied") is True
        assert "mle_source_level" in rates[0].metadata

    def test_compute_batting_rates_skips_mle_for_high_mlb_pa(self) -> None:
        """Players with sufficient MLB PA should not use MLE."""
        from fantasy_baseball_manager.minors.features import MLEBatterFeatureExtractor
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputer

        extractor = MLEBatterFeatureExtractor(min_pa=50)
        feature_names = extractor.feature_names()
        mle_model = _create_fitted_mle_model(feature_names)

        # Player with high MLB PA (500)
        mlb_player = _make_player(
            player_id="654321",
            year=2024,
            age=28,
            pa=500,  # Above threshold - established MLB player
        )
        league = _make_league(year=2024)

        mlb_source = FakeMLBDataSource(
            player_batting={2024: [mlb_player], 2023: [], 2022: []},
            team_batting_data={
                2024: [league],
                2023: [_make_league(2023)],
                2022: [_make_league(2022)],
            },
        )

        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store.load.return_value = mle_model

        computer = MLERateComputer(
            milb_source=MagicMock(),
            model_store=mock_store,
        )

        rates = computer.compute_batting_rates(mlb_source, 2025, 3)

        assert len(rates) == 1
        assert rates[0].player_id == "654321"
        # MLE should NOT be applied
        assert rates[0].metadata.get("mle_applied") is not True

    def test_compute_pitching_rates_falls_back_to_marcel(self) -> None:
        """Pitching rates should fall back to Marcel (MLE pitchers not implemented)."""
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputer

        pitcher = _make_pitcher(player_id="sp1", year=2024)
        league = _make_league_pitching(year=2024)

        mlb_source = FakeMLBDataSource(
            player_pitching={2024: [pitcher], 2023: [], 2022: []},
            team_pitching_data={
                2024: [league],
                2023: [_make_league_pitching(2023)],
                2022: [_make_league_pitching(2022)],
            },
        )

        computer = MLERateComputer(
            milb_source=MagicMock(),
            model_store=MagicMock(),
        )

        rates = computer.compute_pitching_rates(mlb_source, 2025, 3)

        assert len(rates) == 1
        assert rates[0].player_id == "sp1"
        # Should have Marcel metadata, not MLE
        assert rates[0].metadata.get("mle_applied") is not True

    def test_blends_mlb_and_mle_rates(self) -> None:
        """MLE rates should be blended with MLB rates based on PA."""
        from fantasy_baseball_manager.minors.features import MLEBatterFeatureExtractor
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputer

        extractor = MLEBatterFeatureExtractor(min_pa=50)
        feature_names = extractor.feature_names()
        mle_model = _create_fitted_mle_model(feature_names)

        # Player with some MLB PA
        mlb_player = _make_player(
            player_id="123456",
            year=2024,
            pa=100,  # Some MLB experience
            hr=5,  # 0.05 HR rate
        )
        league = _make_league(year=2024)

        mlb_source = FakeMLBDataSource(
            player_batting={
                2024: [mlb_player],
                2023: [],
                2022: [],
            },
            team_batting_data={
                2024: [league],
                2023: [_make_league(2023)],
                2022: [_make_league(2022)],
            },
        )

        milb_player = _make_milb_batter(
            player_id="123456",
            season=2024,
            pa=400,
        )
        milb_source = FakeMinorLeagueDataSource(batting_data={2024: [milb_player]})

        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store.load.return_value = mle_model

        computer = MLERateComputer(
            milb_source=milb_source,
            model_store=mock_store,
        )

        rates = computer.compute_batting_rates(mlb_source, 2025, 3)

        # Should have blended rates
        assert len(rates) == 1
        assert "hr" in rates[0].rates
        # Marcel rates should be preserved in metadata
        assert "marcel_rates" in rates[0].metadata


class TestMLERateComputerIntegration:
    """Integration tests that verify the full pipeline works."""

    def test_protocol_compliance(self) -> None:
        """MLERateComputer should satisfy RateComputer protocol."""
        from fantasy_baseball_manager.minors.rate_computer import MLERateComputer
        from fantasy_baseball_manager.pipeline.protocols import RateComputer

        computer = MLERateComputer(
            milb_source=MagicMock(),
            model_store=MagicMock(),
        )

        # Check that it has the required methods
        assert hasattr(computer, "compute_batting_rates")
        assert hasattr(computer, "compute_pitching_rates")
        assert callable(computer.compute_batting_rates)
        assert callable(computer.compute_pitching_rates)
