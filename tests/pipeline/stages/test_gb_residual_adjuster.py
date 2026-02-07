"""Tests for gradient boosting residual adjuster."""

from pathlib import Path

import numpy as np
import pytest

from fantasy_baseball_manager.ml.persistence import ModelStore
from fantasy_baseball_manager.ml.residual_model import ResidualModelSet, StatResidualModel
from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats, PitcherSkillStats
from fantasy_baseball_manager.pipeline.stages.gb_residual_adjuster import (
    GBResidualAdjuster,
    GBResidualConfig,
)
from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats, StatcastPitcherStats
from fantasy_baseball_manager.pipeline.types import PlayerRates
from fantasy_baseball_manager.player.identity import Player


class FakeStatcastSource:
    """Fake Statcast source for testing."""

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
    """Fake batted ball source for testing."""

    def __init__(self, stats: dict[int, list[PitcherBattedBallStats]]) -> None:
        self._stats = stats

    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]:
        return self._stats.get(year, [])


class FakeSkillDataSource:
    """Fake skill data source for testing."""

    def __init__(
        self,
        batter_stats: dict[int, list[BatterSkillStats]] | None = None,
        pitcher_stats: dict[int, list[PitcherSkillStats]] | None = None,
    ) -> None:
        self._batter = batter_stats or {}
        self._pitcher = pitcher_stats or {}

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]:
        return self._batter.get(year, [])

    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]:
        return self._pitcher.get(year, [])


def _make_batter(
    player_id: str = "fg123",
    year: int = 2024,
    mlbam_id: str | None = "mlbam123",
) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name="Test Batter",
        year=year,
        age=28,
        rates={
            "hr": 0.040,
            "so": 0.200,
            "bb": 0.100,
            "singles": 0.150,
            "doubles": 0.050,
            "triples": 0.005,
            "sb": 0.020,
        },
        opportunities=500.0,
        metadata={"pa_per_year": [500.0]},
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id=mlbam_id, name="Test Batter"),
    )


def _make_pitcher(
    player_id: str = "fg456",
    year: int = 2024,
    mlbam_id: str | None = "mlbam456",
) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name="Test Pitcher",
        year=year,
        age=30,
        rates={
            "h": 0.080,
            "er": 0.030,
            "so": 0.090,
            "bb": 0.030,
            "hr": 0.010,
        },
        opportunities=600.0,
        metadata={"is_starter": True},
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id=mlbam_id, name="Test Pitcher"),
    )


@pytest.fixture
def temp_model_store(tmp_path: Path) -> ModelStore:
    """Create a model store in a temp directory."""
    return ModelStore(model_dir=tmp_path / "models")


@pytest.fixture
def trained_batter_models(temp_model_store: ModelStore) -> ModelStore:
    """Create and save trained batter models."""
    np.random.seed(42)
    feature_names = [
        "marcel_hr",
        "marcel_so",
        "marcel_bb",
        "marcel_singles",
        "marcel_doubles",
        "marcel_triples",
        "marcel_sb",
        "xba",
        "xslg",
        "xwoba",
        "barrel_rate",
        "hard_hit_rate",
        "chase_rate",
        "whiff_rate",
        "chase_minus_league_avg",
        "whiff_minus_league_avg",
        "chase_x_whiff",
        "discipline_score",
        "has_skill_data",
        "age",
        "age_squared",
        "marcel_iso",
        "xba_minus_marcel_avg",
        "barrel_vs_hr_ratio",
        "opportunities",
    ]

    X = np.random.randn(50, len(feature_names))
    model_set = ResidualModelSet(
        player_type="batter",
        feature_names=feature_names,
        training_years=(2021, 2022),
    )

    for stat in ["hr", "so", "bb"]:
        y = np.random.randn(50) * 2  # Small residuals
        model = StatResidualModel(stat_name=stat)
        model.fit(X, y, feature_names)
        model_set.add_model(model)

    temp_model_store.save(model_set, "default")
    return temp_model_store


class TestGBResidualAdjuster:
    def test_passes_through_when_no_models(self, tmp_path: Path) -> None:
        """Test that players pass through unchanged when models don't exist."""
        adjuster = GBResidualAdjuster(
            statcast_source=FakeStatcastSource({}, {}),
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
            config=GBResidualConfig(model_name="nonexistent"),
            model_store=ModelStore(model_dir=tmp_path / "empty_models"),
        )

        batter = _make_batter()
        result = adjuster.adjust([batter])

        assert len(result) == 1
        assert result[0].rates == batter.rates

    def test_passes_through_when_no_statcast(
        self,
        trained_batter_models: ModelStore,
    ) -> None:
        """Test that batters without Statcast data pass through unchanged."""
        adjuster = GBResidualAdjuster(
            statcast_source=FakeStatcastSource({}, {}),  # No Statcast data
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
            model_store=trained_batter_models,
        )

        batter = _make_batter()
        result = adjuster.adjust([batter])

        assert len(result) == 1
        assert result[0].rates == batter.rates

    def test_passes_through_when_no_id_mapping(
        self,
        trained_batter_models: ModelStore,
    ) -> None:
        """Test that batters without ID mapping pass through unchanged."""
        statcast = StatcastBatterStats(
            player_id="mlbam123",
            name="Test",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        adjuster = GBResidualAdjuster(
            statcast_source=FakeStatcastSource({2023: [statcast]}, {}),
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
            model_store=trained_batter_models,
        )

        batter = _make_batter(mlbam_id=None)  # No MLBAM mapping
        result = adjuster.adjust([batter])

        assert len(result) == 1
        assert result[0].rates == batter.rates

    def test_adjusts_batter_with_models(
        self,
        trained_batter_models: ModelStore,
    ) -> None:
        """Test that batters are adjusted when models exist."""
        statcast = StatcastBatterStats(
            player_id="mlbam123",
            name="Test",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        adjuster = GBResidualAdjuster(
            statcast_source=FakeStatcastSource({2023: [statcast]}, {}),
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
            model_store=trained_batter_models,
        )

        batter = _make_batter()
        result = adjuster.adjust([batter])

        assert len(result) == 1
        # Rates should be different (adjusted)
        assert result[0].rates != batter.rates
        # Should have adjustment metadata
        assert "gb_residual_adjustments" in result[0].metadata

    def test_skips_batter_below_min_pa(
        self,
        trained_batter_models: ModelStore,
    ) -> None:
        """Test that batters with low PA Statcast data are skipped."""
        statcast = StatcastBatterStats(
            player_id="mlbam123",
            name="Test",
            year=2023,
            pa=50,  # Below default min_pa of 100
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        adjuster = GBResidualAdjuster(
            statcast_source=FakeStatcastSource({2023: [statcast]}, {}),
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
            model_store=trained_batter_models,
        )

        batter = _make_batter()
        result = adjuster.adjust([batter])

        assert len(result) == 1
        assert result[0].rates == batter.rates

    def test_caches_data_for_same_year(
        self,
        trained_batter_models: ModelStore,
    ) -> None:
        """Test that Statcast data is cached for multiple players in same year."""
        statcast = StatcastBatterStats(
            player_id="mlbam123",
            name="Test",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        call_count = 0
        original_method = FakeStatcastSource.batter_expected_stats

        def counting_method(self: FakeStatcastSource, year: int) -> list[StatcastBatterStats]:
            nonlocal call_count
            call_count += 1
            return original_method(self, year)

        FakeStatcastSource.batter_expected_stats = counting_method  # type: ignore[method-assign]

        try:
            source = FakeStatcastSource({2023: [statcast]}, {})
            adjuster = GBResidualAdjuster(
                statcast_source=source,
                batted_ball_source=FakeBattedBallSource({}),
                skill_data_source=FakeSkillDataSource(),
                model_store=trained_batter_models,
            )

            batter1 = _make_batter(player_id="fg123")
            batter2 = _make_batter(player_id="fg124")

            # First call loads data
            adjuster.adjust([batter1])
            assert call_count == 1

            # Second call uses cache
            adjuster.adjust([batter2])
            assert call_count == 1  # Still 1, not reloaded
        finally:
            FakeStatcastSource.batter_expected_stats = original_method

    def test_empty_list_returns_empty(self) -> None:
        adjuster = GBResidualAdjuster(
            statcast_source=FakeStatcastSource({}, {}),
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
        )

        result = adjuster.adjust([])
        assert result == []


class TestGBResidualConfig:
    def test_default_values(self) -> None:
        config = GBResidualConfig()
        assert config.model_name == "default"
        assert config.batter_min_pa == 100
        assert config.pitcher_min_pa == 100
        assert config.min_rate_denominator_pa == 300
        assert config.min_rate_denominator_ip == 100

    def test_custom_values(self) -> None:
        config = GBResidualConfig(
            model_name="custom",
            batter_min_pa=200,
            pitcher_min_pa=150,
            min_rate_denominator_pa=400,
            min_rate_denominator_ip=120,
        )
        assert config.model_name == "custom"
        assert config.batter_min_pa == 200
        assert config.pitcher_min_pa == 150
        assert config.min_rate_denominator_pa == 400
        assert config.min_rate_denominator_ip == 120
