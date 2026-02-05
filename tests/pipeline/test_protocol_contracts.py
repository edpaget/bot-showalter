"""Contract tests for pipeline protocol implementations.

These tests verify that all implementations of each protocol satisfy the
protocol's contract correctly. They use parametrized fixtures to run the
same tests against all implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.pipeline.stages.adjusters import (
    MarcelAgingAdjuster,
    RebaselineAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.component_aging import ComponentAgingAdjuster
from fantasy_baseball_manager.pipeline.stages.enhanced_playing_time import EnhancedPlayingTimeProjector
from fantasy_baseball_manager.pipeline.stages.finalizers import StandardFinalizer
from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import PitcherNormalizationAdjuster
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import StatSpecificRegressionRateComputer
from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.protocols import (
        PlayingTimeProjector,
        ProjectionFinalizer,
        RateAdjuster,
        RateComputer,
    )

# =============================================================================
# Fixtures for creating test data
# =============================================================================


@pytest.fixture
def sample_batter_rates() -> PlayerRates:
    """Create a sample batter PlayerRates for testing."""
    # Define league average rates for all stats
    league_rates = {
        "hr": 0.030,
        "bb": 0.080,
        "so": 0.200,
        "singles": 0.150,
        "doubles": 0.045,
        "triples": 0.005,
        "hbp": 0.010,
        "sf": 0.008,
        "sh": 0.002,
        "sb": 0.020,
        "cs": 0.005,
        "r": 0.120,
        "rbi": 0.110,
        "h": 0.235,
    }
    return PlayerRates(
        player_id="b001",
        name="Test Batter",
        year=2025,
        age=28,
        rates={
            "hr": 0.035,
            "bb": 0.085,
            "so": 0.200,
            "singles": 0.150,
            "doubles": 0.045,
            "triples": 0.005,
            "hbp": 0.010,
            "sf": 0.008,
            "sh": 0.002,
            "sb": 0.020,
            "cs": 0.005,
            "r": 0.120,
            "rbi": 0.110,
            "h": 0.235,
        },
        opportunities=550.0,
        metadata={
            "pa_per_year": [500.0, 520.0, 550.0],
            "position": "OF",
            "team": "NYY",
            "avg_league_rates": league_rates,
            "target_rates": league_rates,
        },
    )


@pytest.fixture
def sample_pitcher_rates() -> PlayerRates:
    """Create a sample pitcher PlayerRates for testing."""
    # Define league average rates for all stats
    league_rates = {
        "h": 0.240,
        "hr": 0.030,
        "bb": 0.070,
        "so": 0.250,
        "er": 0.100,
        "hbp": 0.008,
        "w": 0.015,
        "sv": 0.0,
        "hld": 0.0,
    }
    return PlayerRates(
        player_id="p001",
        name="Test Pitcher",
        year=2025,
        age=27,
        rates={
            "h": 0.240,
            "hr": 0.030,
            "bb": 0.070,
            "so": 0.250,
            "er": 0.100,
            "hbp": 0.008,
            "w": 0.015,
            "sv": 0.0,
            "hld": 0.0,
        },
        opportunities=180.0,
        metadata={
            "ip_per_year": [150.0, 170.0, 180.0],
            "is_starter": True,
            "team": "LAD",
            "avg_league_rates": league_rates,
            "target_rates": league_rates,
        },
    )


@pytest.fixture
def sample_batter_list(sample_batter_rates: PlayerRates) -> list[PlayerRates]:
    """Create a list of batter rates for testing."""
    return [sample_batter_rates]


@pytest.fixture
def sample_pitcher_list(sample_pitcher_rates: PlayerRates) -> list[PlayerRates]:
    """Create a list of pitcher rates for testing."""
    return [sample_pitcher_rates]


@pytest.fixture
def sample_mixed_list(
    sample_batter_rates: PlayerRates,
    sample_pitcher_rates: PlayerRates,
) -> list[PlayerRates]:
    """Create a mixed list of batter and pitcher rates for testing."""
    return [sample_batter_rates, sample_pitcher_rates]


# =============================================================================
# RateAdjuster Protocol Contract Tests
# =============================================================================

# RateAdjuster implementations that can be instantiated without constructor arguments.
# Adjusters that require dependencies (StatcastRateAdjuster, BatterBabipAdjuster,
# ParkFactorAdjuster, GBResidualAdjuster, PitcherBabipSkillAdjuster,
# PitcherStatcastAdjuster, SkillChangeAdjuster) are tested in their respective
# unit test files with proper mocks.
RATE_ADJUSTERS: list[type[RateAdjuster]] = [
    RebaselineAdjuster,
    MarcelAgingAdjuster,
    PitcherNormalizationAdjuster,
    ComponentAgingAdjuster,
]


@pytest.fixture(params=RATE_ADJUSTERS, ids=lambda cls: cls.__name__)
def rate_adjuster(request: pytest.FixtureRequest) -> RateAdjuster:
    """Parametrized fixture that yields each RateAdjuster implementation."""
    return request.param()


class TestRateAdjusterContract:
    """Contract tests for RateAdjuster protocol implementations."""

    def test_has_adjust_method(self, rate_adjuster: RateAdjuster) -> None:
        """Verify the implementation has an adjust method."""
        assert hasattr(rate_adjuster, "adjust")
        assert callable(rate_adjuster.adjust)

    def test_adjust_returns_list(
        self,
        rate_adjuster: RateAdjuster,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify adjust returns a list."""
        result = rate_adjuster.adjust(sample_mixed_list)
        assert isinstance(result, list)

    def test_adjust_returns_player_rates(
        self,
        rate_adjuster: RateAdjuster,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify adjust returns PlayerRates objects."""
        result = rate_adjuster.adjust(sample_mixed_list)
        for item in result:
            assert isinstance(item, PlayerRates)

    def test_adjust_preserves_player_count(
        self,
        rate_adjuster: RateAdjuster,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify adjust returns the same number of players."""
        result = rate_adjuster.adjust(sample_mixed_list)
        assert len(result) == len(sample_mixed_list)

    def test_adjust_preserves_player_ids(
        self,
        rate_adjuster: RateAdjuster,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify adjust preserves player IDs."""
        result = rate_adjuster.adjust(sample_mixed_list)
        input_ids = {p.player_id for p in sample_mixed_list}
        output_ids = {p.player_id for p in result}
        assert input_ids == output_ids

    def test_adjust_handles_empty_list(self, rate_adjuster: RateAdjuster) -> None:
        """Verify adjust handles an empty list correctly."""
        result = rate_adjuster.adjust([])
        assert result == []


# =============================================================================
# PlayingTimeProjector Protocol Contract Tests
# =============================================================================

PLAYING_TIME_PROJECTORS: list[type[PlayingTimeProjector]] = [
    EnhancedPlayingTimeProjector,
]


@pytest.fixture(params=PLAYING_TIME_PROJECTORS, ids=lambda cls: cls.__name__)
def playing_time_projector(request: pytest.FixtureRequest) -> PlayingTimeProjector:
    """Parametrized fixture that yields each PlayingTimeProjector implementation."""
    return request.param()


class TestPlayingTimeProjectorContract:
    """Contract tests for PlayingTimeProjector protocol implementations."""

    def test_has_project_method(self, playing_time_projector: PlayingTimeProjector) -> None:
        """Verify the implementation has a project method."""
        assert hasattr(playing_time_projector, "project")
        assert callable(playing_time_projector.project)

    def test_project_returns_list(
        self,
        playing_time_projector: PlayingTimeProjector,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify project returns a list."""
        result = playing_time_projector.project(sample_mixed_list)
        assert isinstance(result, list)

    def test_project_returns_player_rates(
        self,
        playing_time_projector: PlayingTimeProjector,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify project returns PlayerRates objects."""
        result = playing_time_projector.project(sample_mixed_list)
        for item in result:
            assert isinstance(item, PlayerRates)

    def test_project_preserves_player_count(
        self,
        playing_time_projector: PlayingTimeProjector,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify project returns the same number of players."""
        result = playing_time_projector.project(sample_mixed_list)
        assert len(result) == len(sample_mixed_list)

    def test_project_preserves_player_ids(
        self,
        playing_time_projector: PlayingTimeProjector,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify project preserves player IDs."""
        result = playing_time_projector.project(sample_mixed_list)
        input_ids = {p.player_id for p in sample_mixed_list}
        output_ids = {p.player_id for p in result}
        assert input_ids == output_ids

    def test_project_handles_empty_list(self, playing_time_projector: PlayingTimeProjector) -> None:
        """Verify project handles an empty list correctly."""
        result = playing_time_projector.project([])
        assert result == []

    def test_project_sets_opportunities(
        self,
        playing_time_projector: PlayingTimeProjector,
        sample_mixed_list: list[PlayerRates],
    ) -> None:
        """Verify project sets opportunities on output players."""
        result = playing_time_projector.project(sample_mixed_list)
        for item in result:
            # Opportunities should be a non-negative number
            assert isinstance(item.opportunities, (int, float))
            assert item.opportunities >= 0


# =============================================================================
# ProjectionFinalizer Protocol Contract Tests
# =============================================================================

PROJECTION_FINALIZERS: list[type[ProjectionFinalizer]] = [
    StandardFinalizer,
]


@pytest.fixture(params=PROJECTION_FINALIZERS, ids=lambda cls: cls.__name__)
def projection_finalizer(request: pytest.FixtureRequest) -> ProjectionFinalizer:
    """Parametrized fixture that yields each ProjectionFinalizer implementation."""
    return request.param()


class TestProjectionFinalizerContract:
    """Contract tests for ProjectionFinalizer protocol implementations."""

    def test_has_finalize_batting_method(self, projection_finalizer: ProjectionFinalizer) -> None:
        """Verify the implementation has a finalize_batting method."""
        assert hasattr(projection_finalizer, "finalize_batting")
        assert callable(projection_finalizer.finalize_batting)

    def test_has_finalize_pitching_method(self, projection_finalizer: ProjectionFinalizer) -> None:
        """Verify the implementation has a finalize_pitching method."""
        assert hasattr(projection_finalizer, "finalize_pitching")
        assert callable(projection_finalizer.finalize_pitching)

    def test_finalize_batting_returns_list(
        self,
        projection_finalizer: ProjectionFinalizer,
        sample_batter_list: list[PlayerRates],
    ) -> None:
        """Verify finalize_batting returns a list."""
        result = projection_finalizer.finalize_batting(sample_batter_list)
        assert isinstance(result, list)

    def test_finalize_batting_returns_batting_projections(
        self,
        projection_finalizer: ProjectionFinalizer,
        sample_batter_list: list[PlayerRates],
    ) -> None:
        """Verify finalize_batting returns BattingProjection objects."""
        from fantasy_baseball_manager.marcel.models import BattingProjection

        result = projection_finalizer.finalize_batting(sample_batter_list)
        for item in result:
            assert isinstance(item, BattingProjection)

    def test_finalize_pitching_returns_list(
        self,
        projection_finalizer: ProjectionFinalizer,
        sample_pitcher_list: list[PlayerRates],
    ) -> None:
        """Verify finalize_pitching returns a list."""
        result = projection_finalizer.finalize_pitching(sample_pitcher_list)
        assert isinstance(result, list)

    def test_finalize_pitching_returns_pitching_projections(
        self,
        projection_finalizer: ProjectionFinalizer,
        sample_pitcher_list: list[PlayerRates],
    ) -> None:
        """Verify finalize_pitching returns PitchingProjection objects."""
        from fantasy_baseball_manager.marcel.models import PitchingProjection

        result = projection_finalizer.finalize_pitching(sample_pitcher_list)
        for item in result:
            assert isinstance(item, PitchingProjection)

    def test_finalize_batting_preserves_player_count(
        self,
        projection_finalizer: ProjectionFinalizer,
        sample_batter_list: list[PlayerRates],
    ) -> None:
        """Verify finalize_batting returns the same number of players."""
        result = projection_finalizer.finalize_batting(sample_batter_list)
        assert len(result) == len(sample_batter_list)

    def test_finalize_pitching_preserves_player_count(
        self,
        projection_finalizer: ProjectionFinalizer,
        sample_pitcher_list: list[PlayerRates],
    ) -> None:
        """Verify finalize_pitching returns the same number of players."""
        result = projection_finalizer.finalize_pitching(sample_pitcher_list)
        assert len(result) == len(sample_pitcher_list)

    def test_finalize_batting_handles_empty_list(
        self,
        projection_finalizer: ProjectionFinalizer,
    ) -> None:
        """Verify finalize_batting handles an empty list correctly."""
        result = projection_finalizer.finalize_batting([])
        assert result == []

    def test_finalize_pitching_handles_empty_list(
        self,
        projection_finalizer: ProjectionFinalizer,
    ) -> None:
        """Verify finalize_pitching handles an empty list correctly."""
        result = projection_finalizer.finalize_pitching([])
        assert result == []

    def test_finalize_batting_preserves_player_ids(
        self,
        projection_finalizer: ProjectionFinalizer,
        sample_batter_list: list[PlayerRates],
    ) -> None:
        """Verify finalize_batting preserves player IDs."""
        result = projection_finalizer.finalize_batting(sample_batter_list)
        input_ids = {p.player_id for p in sample_batter_list}
        output_ids = {p.player_id for p in result}
        assert input_ids == output_ids

    def test_finalize_pitching_preserves_player_ids(
        self,
        projection_finalizer: ProjectionFinalizer,
        sample_pitcher_list: list[PlayerRates],
    ) -> None:
        """Verify finalize_pitching preserves player IDs."""
        result = projection_finalizer.finalize_pitching(sample_pitcher_list)
        input_ids = {p.player_id for p in sample_pitcher_list}
        output_ids = {p.player_id for p in result}
        assert input_ids == output_ids


# =============================================================================
# RateComputer Protocol Contract Tests
# =============================================================================

# Simple rate computers that don't require additional dependencies
RATE_COMPUTERS_SIMPLE: list[type[RateComputer]] = [
    MarcelRateComputer,
    StatSpecificRegressionRateComputer,
]


@pytest.fixture(params=RATE_COMPUTERS_SIMPLE, ids=lambda cls: cls.__name__)
def rate_computer(request: pytest.FixtureRequest) -> RateComputer:
    """Parametrized fixture that yields each simple RateComputer implementation."""
    return request.param()


class TestRateComputerContract:
    """Contract tests for RateComputer protocol implementations.

    Note: PlatoonRateComputer is tested separately as it requires a
    SplitStatsDataSource and a delegate RateComputer.

    Note: Full behavioral tests with actual data are in individual unit test files.
    These contract tests verify protocol compliance (method signatures).
    """

    def test_has_compute_batting_rates_method(self, rate_computer: RateComputer) -> None:
        """Verify the implementation has a compute_batting_rates method."""
        assert hasattr(rate_computer, "compute_batting_rates")
        assert callable(rate_computer.compute_batting_rates)

    def test_has_compute_pitching_rates_method(self, rate_computer: RateComputer) -> None:
        """Verify the implementation has a compute_pitching_rates method."""
        assert hasattr(rate_computer, "compute_pitching_rates")
        assert callable(rate_computer.compute_pitching_rates)

    def test_compute_batting_rates_accepts_correct_signature(
        self,
        rate_computer: RateComputer,
    ) -> None:
        """Verify compute_batting_rates accepts the protocol signature."""
        import inspect

        sig = inspect.signature(rate_computer.compute_batting_rates)
        params = list(sig.parameters.keys())
        # Should have data_source, year, years_back (possibly with self)
        assert "data_source" in params or len(params) >= 3

    def test_compute_pitching_rates_accepts_correct_signature(
        self,
        rate_computer: RateComputer,
    ) -> None:
        """Verify compute_pitching_rates accepts the protocol signature."""
        import inspect

        sig = inspect.signature(rate_computer.compute_pitching_rates)
        params = list(sig.parameters.keys())
        # Should have data_source, year, years_back (possibly with self)
        assert "data_source" in params or len(params) >= 3
