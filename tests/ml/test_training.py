"""Tests for model training orchestration."""

from dataclasses import dataclass

import numpy as np
import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.ml.training import (
    BATTER_STATS,
    PITCHER_STATS,
    ResidualModelTrainer,
    TrainingConfig,
)
from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats, StatcastPitcherStats


class FakeDataSource:
    """Fake data source for testing."""

    def __init__(
        self,
        batting_stats: dict[int, list[BattingSeasonStats]],
        pitching_stats: dict[int, list[PitchingSeasonStats]],
    ) -> None:
        self._batting = batting_stats
        self._pitching = pitching_stats

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return []

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return []


class FakeStatcastSource:
    """Fake Statcast data source for testing."""

    def __init__(
        self,
        batter_stats: dict[int, list[StatcastBatterStats]],
        pitcher_stats: dict[int, list[StatcastPitcherStats]],
    ) -> None:
        self._batter = batter_stats
        self._pitcher = pitcher_stats

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return self._batter.get(year, [])

    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]:
        return self._pitcher.get(year, [])


class FakeBattedBallSource:
    """Fake batted ball data source for testing."""

    def __init__(self, stats: dict[int, list[PitcherBattedBallStats]]) -> None:
        self._stats = stats

    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]:
        return self._stats.get(year, [])


class FakeIdMapper:
    """Fake ID mapper for testing."""

    def __init__(self, fg_to_mlbam: dict[str, str]) -> None:
        self._fg_to_mlbam = fg_to_mlbam
        self._mlbam_to_fg = {v: k for k, v in fg_to_mlbam.items()}

    def fangraphs_to_mlbam(self, fg_id: str) -> str | None:
        return self._fg_to_mlbam.get(fg_id)

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        return self._mlbam_to_fg.get(mlbam_id)

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return None

    def fangraphs_to_yahoo(self, fg_id: str) -> str | None:
        return None


class FakePipeline:
    """Fake pipeline that returns predefined projections."""

    def __init__(
        self,
        batting_projections: dict[int, list[BattingProjection]],
        pitching_projections: dict[int, list[PitchingProjection]],
    ) -> None:
        self._batting = batting_projections
        self._pitching = pitching_projections

    def project_batters(self, data_source: object, year: int) -> list[BattingProjection]:
        return self._batting.get(year, [])

    def project_pitchers(self, data_source: object, year: int) -> list[PitchingProjection]:
        return self._pitching.get(year, [])


def _make_batter_projection(player_id: str, year: int) -> BattingProjection:
    return BattingProjection(
        player_id=player_id,
        name=f"Player {player_id}",
        year=year,
        age=28,
        pa=500.0,
        ab=450.0,
        h=120.0,
        singles=80.0,
        doubles=25.0,
        triples=3.0,
        hr=20.0,
        bb=50.0,
        so=100.0,
        hbp=5.0,
        sf=4.0,
        sh=1.0,
        sb=10.0,
        cs=3.0,
        r=70.0,
        rbi=65.0,
    )


def _make_batter_actuals(player_id: str, year: int) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=f"Player {player_id}",
        year=year,
        age=28,
        pa=520,
        ab=470,
        h=130,
        singles=85,
        doubles=28,
        triples=4,
        hr=22,  # Slightly different from projection
        bb=55,
        so=95,
        hbp=5,
        sf=4,
        sh=1,
        sb=12,
        cs=4,
        r=75,
        rbi=70,
    )


def _make_statcast_batter(mlbam_id: str, year: int) -> StatcastBatterStats:
    return StatcastBatterStats(
        player_id=mlbam_id,
        name=f"Player {mlbam_id}",
        year=year,
        pa=450,
        barrel_rate=0.08,
        hard_hit_rate=0.40,
        xwoba=0.350,
        xba=0.280,
        xslg=0.450,
    )


class TestResidualModelTrainer:
    def test_train_batter_models_with_sufficient_data(self) -> None:
        """Test training batter models with enough samples."""
        # Create data for multiple years and players
        players = [f"fg{i}" for i in range(60)]
        fg_to_mlbam = {f"fg{i}": f"mlbam{i}" for i in range(60)}

        batting_projections: dict[int, list[BattingProjection]] = {}
        batting_actuals: dict[int, list[BattingSeasonStats]] = {}
        statcast_batter: dict[int, list[StatcastBatterStats]] = {}

        for year in [2021, 2022]:
            batting_projections[year] = [_make_batter_projection(p, year) for p in players]
            batting_actuals[year] = [_make_batter_actuals(p, year) for p in players]
            statcast_batter[year - 1] = [
                _make_statcast_batter(f"mlbam{i}", year - 1) for i in range(60)
            ]

        trainer = ResidualModelTrainer(
            pipeline=FakePipeline(batting_projections, {}),  # type: ignore[arg-type]
            data_source=FakeDataSource(batting_actuals, {}),  # type: ignore[arg-type]
            statcast_source=FakeStatcastSource(statcast_batter, {}),  # type: ignore[arg-type]
            batted_ball_source=FakeBattedBallSource({}),  # type: ignore[arg-type]
            id_mapper=FakeIdMapper(fg_to_mlbam),  # type: ignore[arg-type]
            config=TrainingConfig(min_samples=50, batter_min_pa=100),
        )

        model_set = trainer.train_batter_models((2021, 2022))

        assert model_set.player_type == "batter"
        assert model_set.training_years == (2021, 2022)
        # Should have trained models for stats with enough samples
        stats = model_set.get_stats()
        assert len(stats) > 0
        for stat in stats:
            assert stat in BATTER_STATS

    def test_train_batter_models_skips_insufficient_data(self) -> None:
        """Test that training skips stats with insufficient samples."""
        # Only 10 players - below min_samples
        players = [f"fg{i}" for i in range(10)]
        fg_to_mlbam = {f"fg{i}": f"mlbam{i}" for i in range(10)}

        batting_projections = {2022: [_make_batter_projection(p, 2022) for p in players]}
        batting_actuals = {2022: [_make_batter_actuals(p, 2022) for p in players]}
        statcast_batter = {2021: [_make_statcast_batter(f"mlbam{i}", 2021) for i in range(10)]}

        trainer = ResidualModelTrainer(
            pipeline=FakePipeline(batting_projections, {}),  # type: ignore[arg-type]
            data_source=FakeDataSource(batting_actuals, {}),  # type: ignore[arg-type]
            statcast_source=FakeStatcastSource(statcast_batter, {}),  # type: ignore[arg-type]
            batted_ball_source=FakeBattedBallSource({}),  # type: ignore[arg-type]
            id_mapper=FakeIdMapper(fg_to_mlbam),  # type: ignore[arg-type]
            config=TrainingConfig(min_samples=50),  # Require 50 samples
        )

        model_set = trainer.train_batter_models((2022,))

        # Should have no trained models due to insufficient samples
        assert len(model_set.get_stats()) == 0

    def test_train_batter_models_skips_unmapped_players(self) -> None:
        """Test that players without ID mapping are skipped."""
        players = [f"fg{i}" for i in range(60)]
        # Only map half the players
        fg_to_mlbam = {f"fg{i}": f"mlbam{i}" for i in range(30)}

        batting_projections = {2022: [_make_batter_projection(p, 2022) for p in players]}
        batting_actuals = {2022: [_make_batter_actuals(p, 2022) for p in players]}
        # Statcast only for mapped players
        statcast_batter = {2021: [_make_statcast_batter(f"mlbam{i}", 2021) for i in range(30)]}

        trainer = ResidualModelTrainer(
            pipeline=FakePipeline(batting_projections, {}),  # type: ignore[arg-type]
            data_source=FakeDataSource(batting_actuals, {}),  # type: ignore[arg-type]
            statcast_source=FakeStatcastSource(statcast_batter, {}),  # type: ignore[arg-type]
            batted_ball_source=FakeBattedBallSource({}),  # type: ignore[arg-type]
            id_mapper=FakeIdMapper(fg_to_mlbam),  # type: ignore[arg-type]
            config=TrainingConfig(min_samples=20),
        )

        model_set = trainer.train_batter_models((2022,))

        # Should train on the 30 mapped players
        assert len(model_set.get_stats()) > 0


class TestTrainingConfig:
    def test_default_values(self) -> None:
        config = TrainingConfig()
        assert config.min_samples == 50
        assert config.batter_min_pa == 100
        assert config.pitcher_min_pa == 100

    def test_custom_values(self) -> None:
        config = TrainingConfig(
            min_samples=100,
            batter_min_pa=200,
            pitcher_min_pa=150,
        )
        assert config.min_samples == 100
        assert config.batter_min_pa == 200
        assert config.pitcher_min_pa == 150
