"""Contract tests for ROS (rest-of-season) protocol implementations.

These tests verify that all implementations of the ProjectionBlender protocol
satisfy the protocol's contract correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.ros.blender import BayesianBlender

if TYPE_CHECKING:
    from fantasy_baseball_manager.ros.protocol import ProjectionBlender

# =============================================================================
# Fixtures for creating test data
# =============================================================================


@pytest.fixture
def sample_batting_projection() -> BattingProjection:
    """Create a sample BattingProjection for testing."""
    return BattingProjection(
        player_id="b001",
        name="Test Batter",
        year=2025,
        age=28,
        pa=550.0,
        ab=480.0,
        h=140.0,
        singles=90.0,
        doubles=30.0,
        triples=3.0,
        hr=25.0,
        bb=55.0,
        so=110.0,
        hbp=5.0,
        sf=5.0,
        sh=2.0,
        sb=15.0,
        cs=3.0,
        r=80.0,
        rbi=75.0,
    )


@pytest.fixture
def sample_batting_actuals() -> BattingSeasonStats:
    """Create sample BattingSeasonStats for testing."""
    return BattingSeasonStats(
        player_id="b001",
        name="Test Batter",
        year=2025,
        age=28,
        pa=200,
        ab=175,
        h=50,
        singles=32,
        doubles=12,
        triples=1,
        hr=8,
        bb=20,
        so=40,
        hbp=2,
        sf=2,
        sh=1,
        sb=5,
        cs=1,
        r=28,
        rbi=25,
    )


@pytest.fixture
def sample_pitching_projection() -> PitchingProjection:
    """Create a sample PitchingProjection for testing."""
    return PitchingProjection(
        player_id="p001",
        name="Test Pitcher",
        year=2025,
        age=27,
        ip=180.0,
        g=32.0,
        gs=32.0,
        h=160.0,
        hr=20.0,
        bb=50.0,
        so=200.0,
        er=70.0,
        hbp=8.0,
        w=12.0,
        era=3.50,
        whip=1.17,
        nsvh=0.0,
    )


@pytest.fixture
def sample_pitching_actuals() -> PitchingSeasonStats:
    """Create sample PitchingSeasonStats for testing."""
    return PitchingSeasonStats(
        player_id="p001",
        name="Test Pitcher",
        year=2025,
        age=27,
        ip=60.0,
        g=10,
        gs=10,
        h=55,
        hr=7,
        bb=18,
        so=65,
        er=25,
        hbp=3,
        w=4,
        sv=0,
        hld=0,
        bs=1,
    )


# =============================================================================
# ProjectionBlender Protocol Contract Tests
# =============================================================================

PROJECTION_BLENDERS: list[type[ProjectionBlender]] = [
    BayesianBlender,
]


@pytest.fixture(params=PROJECTION_BLENDERS, ids=lambda cls: cls.__name__)
def projection_blender(request: pytest.FixtureRequest) -> ProjectionBlender:
    """Parametrized fixture that yields each ProjectionBlender implementation."""
    return request.param()


class TestProjectionBlenderContract:
    """Contract tests for ProjectionBlender protocol implementations."""

    def test_has_blend_batting_method(self, projection_blender: ProjectionBlender) -> None:
        """Verify the implementation has a blend_batting method."""
        assert hasattr(projection_blender, "blend_batting")
        assert callable(projection_blender.blend_batting)

    def test_has_blend_pitching_method(self, projection_blender: ProjectionBlender) -> None:
        """Verify the implementation has a blend_pitching method."""
        assert hasattr(projection_blender, "blend_pitching")
        assert callable(projection_blender.blend_pitching)

    def test_blend_batting_returns_batting_projection(
        self,
        projection_blender: ProjectionBlender,
        sample_batting_projection: BattingProjection,
        sample_batting_actuals: BattingSeasonStats,
    ) -> None:
        """Verify blend_batting returns a BattingProjection."""
        result = projection_blender.blend_batting(sample_batting_projection, sample_batting_actuals)
        assert isinstance(result, BattingProjection)

    def test_blend_pitching_returns_pitching_projection(
        self,
        projection_blender: ProjectionBlender,
        sample_pitching_projection: PitchingProjection,
        sample_pitching_actuals: PitchingSeasonStats,
    ) -> None:
        """Verify blend_pitching returns a PitchingProjection."""
        result = projection_blender.blend_pitching(sample_pitching_projection, sample_pitching_actuals)
        assert isinstance(result, PitchingProjection)

    def test_blend_batting_preserves_player_id(
        self,
        projection_blender: ProjectionBlender,
        sample_batting_projection: BattingProjection,
        sample_batting_actuals: BattingSeasonStats,
    ) -> None:
        """Verify blend_batting preserves the player ID."""
        result = projection_blender.blend_batting(sample_batting_projection, sample_batting_actuals)
        assert result.player_id == sample_batting_projection.player_id

    def test_blend_pitching_preserves_player_id(
        self,
        projection_blender: ProjectionBlender,
        sample_pitching_projection: PitchingProjection,
        sample_pitching_actuals: PitchingSeasonStats,
    ) -> None:
        """Verify blend_pitching preserves the player ID."""
        result = projection_blender.blend_pitching(sample_pitching_projection, sample_pitching_actuals)
        assert result.player_id == sample_pitching_projection.player_id

    def test_blend_batting_preserves_year(
        self,
        projection_blender: ProjectionBlender,
        sample_batting_projection: BattingProjection,
        sample_batting_actuals: BattingSeasonStats,
    ) -> None:
        """Verify blend_batting preserves the year."""
        result = projection_blender.blend_batting(sample_batting_projection, sample_batting_actuals)
        assert result.year == sample_batting_projection.year

    def test_blend_pitching_preserves_year(
        self,
        projection_blender: ProjectionBlender,
        sample_pitching_projection: PitchingProjection,
        sample_pitching_actuals: PitchingSeasonStats,
    ) -> None:
        """Verify blend_pitching preserves the year."""
        result = projection_blender.blend_pitching(sample_pitching_projection, sample_pitching_actuals)
        assert result.year == sample_pitching_projection.year

    def test_blend_batting_has_non_negative_stats(
        self,
        projection_blender: ProjectionBlender,
        sample_batting_projection: BattingProjection,
        sample_batting_actuals: BattingSeasonStats,
    ) -> None:
        """Verify blend_batting returns non-negative counting stats."""
        result = projection_blender.blend_batting(sample_batting_projection, sample_batting_actuals)
        assert result.pa >= 0
        assert result.ab >= 0
        assert result.h >= 0
        assert result.hr >= 0
        assert result.bb >= 0
        assert result.so >= 0
        assert result.r >= 0
        assert result.rbi >= 0

    def test_blend_pitching_has_non_negative_stats(
        self,
        projection_blender: ProjectionBlender,
        sample_pitching_projection: PitchingProjection,
        sample_pitching_actuals: PitchingSeasonStats,
    ) -> None:
        """Verify blend_pitching returns non-negative counting stats."""
        result = projection_blender.blend_pitching(sample_pitching_projection, sample_pitching_actuals)
        assert result.ip >= 0
        assert result.h >= 0
        assert result.hr >= 0
        assert result.bb >= 0
        assert result.so >= 0
        assert result.er >= 0
        assert result.w >= 0
