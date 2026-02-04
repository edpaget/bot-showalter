"""Tests for minors/training_data.py"""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.marcel.models import BattingSeasonStats
from fantasy_baseball_manager.minors.training_data import (
    BATTER_TARGET_STATS,
    AggregatedMiLBStats,
    MLETrainingDataCollector,
)
from fantasy_baseball_manager.minors.types import (
    MinorLeagueBatterSeasonStats,
    MinorLeagueLevel,
)


def _make_milb_stats(
    player_id: str = "12345",
    name: str = "Test Player",
    season: int = 2023,
    age: int = 23,
    level: MinorLeagueLevel = MinorLeagueLevel.AAA,
    pa: int = 500,
    h: int = 130,
    hr: int = 10,
    so: int = 100,
    bb: int = 40,
    sb: int = 15,
) -> MinorLeagueBatterSeasonStats:
    """Create a MinorLeagueBatterSeasonStats with sensible defaults."""
    singles = h - 25 - 5 - hr  # h - doubles - triples - hr
    return MinorLeagueBatterSeasonStats(
        player_id=player_id,
        name=name,
        season=season,
        age=age,
        level=level,
        team="Test Team",
        league="Test League",
        pa=pa,
        ab=pa - bb - 5,  # approximate
        h=h,
        singles=singles,
        doubles=25,
        triples=5,
        hr=hr,
        rbi=60,
        r=70,
        bb=bb,
        so=so,
        hbp=5,
        sf=5,
        sb=sb,
        cs=5,
        avg=h / (pa - bb - 5) if pa - bb - 5 > 0 else 0.0,
        obp=(h + bb + 5) / pa if pa > 0 else 0.0,
        slg=0.440,
    )


def _make_mlb_stats(
    player_id: str = "12345",
    name: str = "Test Player",
    year: int = 2024,
    age: int = 24,
    pa: int = 300,
    h: int = 80,
    hr: int = 8,
    so: int = 70,
    bb: int = 25,
    sb: int = 8,
) -> BattingSeasonStats:
    """Create a BattingSeasonStats with sensible defaults."""
    singles = h - 15 - 3 - hr
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=pa - bb - 3,
        h=h,
        singles=singles,
        doubles=15,
        triples=3,
        hr=hr,
        bb=bb,
        so=so,
        hbp=3,
        sf=2,
        sh=0,
        sb=sb,
        cs=2,
        r=40,
        rbi=35,
        team="NYM",
    )


class TestAggregatedMiLBStats:
    """Tests for AggregatedMiLBStats."""

    def test_from_single_level_season(self) -> None:
        """Single level season should pass through stats."""
        milb_stats = [_make_milb_stats(pa=500, hr=10, so=100, bb=40)]
        agg = AggregatedMiLBStats.from_seasons(milb_stats)

        assert agg is not None
        assert agg.player_id == "12345"
        assert agg.total_pa == 500
        assert agg.highest_level == MinorLeagueLevel.AAA
        assert agg.pct_at_aaa == 1.0
        assert agg.pct_at_aa == 0.0
        assert agg.hr_rate == 10 / 500
        assert agg.so_rate == 100 / 500
        assert agg.bb_rate == 40 / 500

    def test_from_multi_level_season_pa_weighted(self) -> None:
        """Multi-level season should PA-weight stats."""
        milb_stats = [
            _make_milb_stats(
                level=MinorLeagueLevel.AA, pa=300, hr=12, so=60, bb=30
            ),
            _make_milb_stats(
                level=MinorLeagueLevel.AAA, pa=200, hr=8, so=50, bb=20
            ),
        ]
        agg = AggregatedMiLBStats.from_seasons(milb_stats)

        assert agg is not None
        assert agg.total_pa == 500
        assert agg.highest_level == MinorLeagueLevel.AAA  # Lower sport_id
        assert agg.pct_at_aa == 300 / 500
        assert agg.pct_at_aaa == 200 / 500
        # PA-weighted: (12 + 8) / 500 = 0.04
        assert agg.hr_rate == (12 + 8) / 500
        # PA-weighted: (60 + 50) / 500 = 0.22
        assert agg.so_rate == (60 + 50) / 500

    def test_from_empty_list_returns_none(self) -> None:
        """Empty list should return None."""
        assert AggregatedMiLBStats.from_seasons([]) is None

    def test_from_zero_pa_returns_none(self) -> None:
        """Zero total PA should return None."""
        milb_stats = [_make_milb_stats(pa=0)]
        assert AggregatedMiLBStats.from_seasons(milb_stats) is None

    def test_highest_level_selection(self) -> None:
        """Highest level should be the one with lowest sport_id."""
        milb_stats = [
            _make_milb_stats(level=MinorLeagueLevel.HIGH_A, pa=200),
            _make_milb_stats(level=MinorLeagueLevel.AA, pa=300),
        ]
        agg = AggregatedMiLBStats.from_seasons(milb_stats)

        assert agg is not None
        assert agg.highest_level == MinorLeagueLevel.AA  # AA < HIGH_A

    def test_level_distribution(self) -> None:
        """Level distribution should be calculated correctly."""
        milb_stats = [
            _make_milb_stats(level=MinorLeagueLevel.SINGLE_A, pa=100),
            _make_milb_stats(level=MinorLeagueLevel.HIGH_A, pa=150),
            _make_milb_stats(level=MinorLeagueLevel.AA, pa=200),
            _make_milb_stats(level=MinorLeagueLevel.AAA, pa=50),
        ]
        agg = AggregatedMiLBStats.from_seasons(milb_stats)

        assert agg is not None
        total = 500
        assert agg.pct_at_single_a == 100 / total
        assert agg.pct_at_high_a == 150 / total
        assert agg.pct_at_aa == 200 / total
        assert agg.pct_at_aaa == 50 / total


class _FakeMinorLeagueDataSource:
    """Fake MiLB data source for testing."""

    def __init__(
        self, data: dict[int, list[MinorLeagueBatterSeasonStats]] | None = None
    ) -> None:
        self._data = data or {}

    def batting_stats_all_levels(
        self, year: int
    ) -> list[MinorLeagueBatterSeasonStats]:
        return self._data.get(year, [])


class _FakeMLBDataSource:
    """Fake MLB data source for testing."""

    def __init__(
        self, data: dict[int, list[BattingSeasonStats]] | None = None
    ) -> None:
        self._data = data or {}

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._data.get(year, [])


class TestMLETrainingDataCollector:
    """Tests for MLETrainingDataCollector."""

    def test_collect_qualifying_sample(self) -> None:
        """Collect a qualifying player with sufficient MiLB and MLB PA."""
        milb_source = _FakeMinorLeagueDataSource(
            {
                2023: [
                    _make_milb_stats(
                        player_id="123",
                        pa=250,
                        level=MinorLeagueLevel.AAA,
                        age=23,
                        hr=15,
                        so=60,
                        bb=30,
                    )
                ]
            }
        )
        mlb_source = _FakeMLBDataSource(
            {
                2023: [],  # No prior MLB experience
                2024: [
                    _make_mlb_stats(
                        player_id="123", pa=150, hr=10, so=40, bb=15
                    )
                ],
                2025: [],
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
            min_milb_pa=200,
            min_mlb_pa=100,
            max_prior_mlb_pa=200,
        )

        features, targets, weights, feature_names = collector.collect((2024,))

        assert features.shape[0] == 1
        assert len(feature_names) > 0
        assert "hr" in targets
        assert weights[0] == 150

    def test_excludes_insufficient_milb_pa(self) -> None:
        """Players with insufficient MiLB PA should be excluded."""
        milb_source = _FakeMinorLeagueDataSource(
            {2023: [_make_milb_stats(player_id="123", pa=100, level=MinorLeagueLevel.AAA)]}
        )
        mlb_source = _FakeMLBDataSource(
            {
                2023: [],
                2024: [_make_mlb_stats(player_id="123", pa=200)],
                2025: [],
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
            min_milb_pa=200,  # Requires 200, player has 100
            min_mlb_pa=100,
        )

        features, _targets, _weights, _ = collector.collect((2024,))
        assert features.shape[0] == 0

    def test_excludes_insufficient_mlb_pa(self) -> None:
        """Players with insufficient MLB PA should be excluded."""
        milb_source = _FakeMinorLeagueDataSource(
            {2023: [_make_milb_stats(player_id="123", pa=300, level=MinorLeagueLevel.AAA)]}
        )
        mlb_source = _FakeMLBDataSource(
            {
                2023: [],
                2024: [_make_mlb_stats(player_id="123", pa=50)],  # Below threshold
                2025: [_make_mlb_stats(player_id="123", pa=30)],  # Still below
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
            min_milb_pa=200,
            min_mlb_pa=100,  # Requires 100, player has max 50
        )

        features, _, _, _ = collector.collect((2024,))
        # Player excluded - neither target year nor year+1 has enough PA
        assert features.shape[0] == 0

    def test_excludes_lower_level_players(self) -> None:
        """Players whose highest level is below AA should be excluded."""
        milb_source = _FakeMinorLeagueDataSource(
            {2023: [_make_milb_stats(player_id="123", pa=300, level=MinorLeagueLevel.HIGH_A)]}
        )
        mlb_source = _FakeMLBDataSource(
            {
                2023: [],
                2024: [_make_mlb_stats(player_id="123", pa=200)],
                2025: [],
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
            min_milb_pa=200,
            min_mlb_pa=100,
        )

        features, _targets, _weights, _ = collector.collect((2024,))
        assert features.shape[0] == 0

    def test_excludes_veteran_players(self) -> None:
        """Players with prior MLB experience should be excluded."""
        milb_source = _FakeMinorLeagueDataSource(
            {2023: [_make_milb_stats(player_id="123", pa=300, level=MinorLeagueLevel.AAA)]}
        )
        mlb_source = _FakeMLBDataSource(
            {
                2023: [_make_mlb_stats(player_id="123", pa=300)],  # Prior experience
                2024: [_make_mlb_stats(player_id="123", pa=400)],
                2025: [],
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
            min_milb_pa=200,
            min_mlb_pa=100,
            max_prior_mlb_pa=200,  # Player has 300 PA in prior year
        )

        features, _targets, _weights, _ = collector.collect((2024,))
        assert features.shape[0] == 0

    def test_uses_year_plus_one_for_late_callups(self) -> None:
        """Late-season call-ups should use year+1 MLB stats if more PA."""
        milb_source = _FakeMinorLeagueDataSource(
            {2023: [_make_milb_stats(player_id="123", pa=300, level=MinorLeagueLevel.AAA)]}
        )
        mlb_source = _FakeMLBDataSource(
            {
                2023: [],
                2024: [_make_mlb_stats(player_id="123", pa=50)],  # Below threshold
                2025: [_make_mlb_stats(player_id="123", pa=200)],  # Above threshold
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
            min_milb_pa=200,
            min_mlb_pa=100,
        )

        features, _targets, weights, _ = collector.collect((2024,))
        assert features.shape[0] == 1
        assert weights[0] == 200  # Uses 2025 PA

    def test_feature_extraction(self) -> None:
        """Features should be extracted correctly from aggregated stats."""
        milb_source = _FakeMinorLeagueDataSource(
            {
                2023: [
                    _make_milb_stats(
                        player_id="123",
                        pa=500,
                        level=MinorLeagueLevel.AAA,
                        age=23,
                        hr=20,
                        so=100,
                        bb=50,
                        sb=15,
                    )
                ]
            }
        )
        mlb_source = _FakeMLBDataSource(
            {
                2023: [],
                2024: [_make_mlb_stats(player_id="123", pa=200)],
                2025: [],
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
        )

        features, _targets, _weights, feature_names = collector.collect((2024,))

        assert len(feature_names) == 32  # All feature names (including Statcast)
        assert features.shape == (1, 32)

        # Check some specific features
        hr_idx = feature_names.index("hr_rate")
        assert features[0, hr_idx] == pytest.approx(20 / 500)

        so_idx = feature_names.index("so_rate")
        assert features[0, so_idx] == pytest.approx(100 / 500)

        age_idx = feature_names.index("age")
        assert features[0, age_idx] == 23

    def test_target_extraction(self) -> None:
        """Targets should be MLB rates."""
        milb_source = _FakeMinorLeagueDataSource(
            {2023: [_make_milb_stats(player_id="123", pa=300, level=MinorLeagueLevel.AAA)]}
        )
        mlb_source = _FakeMLBDataSource(
            {
                2023: [],
                2024: [
                    _make_mlb_stats(
                        player_id="123",
                        pa=400,
                        hr=16,
                        so=80,
                        bb=40,
                        sb=10,
                    )
                ],
                2025: [],
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
        )

        _features, targets, _weights, _ = collector.collect((2024,))

        assert targets["hr"][0] == pytest.approx(16 / 400)
        assert targets["so"][0] == pytest.approx(80 / 400)
        assert targets["bb"][0] == pytest.approx(40 / 400)
        assert targets["sb"][0] == pytest.approx(10 / 400)

    def test_collect_multiple_years(self) -> None:
        """Collecting multiple target years should combine samples."""
        milb_source = _FakeMinorLeagueDataSource(
            {
                2022: [_make_milb_stats(player_id="100", pa=300, level=MinorLeagueLevel.AAA)],
                2023: [_make_milb_stats(player_id="200", pa=300, level=MinorLeagueLevel.AAA)],
            }
        )
        mlb_source = _FakeMLBDataSource(
            {
                2022: [],
                2023: [_make_mlb_stats(player_id="100", pa=200)],
                2024: [_make_mlb_stats(player_id="200", pa=200)],
                2025: [],
            }
        )

        collector = MLETrainingDataCollector(
            milb_source=milb_source,  # type: ignore[arg-type]
            mlb_source=mlb_source,  # type: ignore[arg-type]
        )

        features, _targets, _weights, _ = collector.collect((2023, 2024))
        assert features.shape[0] == 2

    def test_feature_names_method(self) -> None:
        """feature_names() should return consistent feature names."""
        collector = MLETrainingDataCollector(
            milb_source=_FakeMinorLeagueDataSource(),  # type: ignore[arg-type]
            mlb_source=_FakeMLBDataSource(),  # type: ignore[arg-type]
        )

        feature_names = collector.feature_names()

        assert len(feature_names) == 32
        assert "hr_rate" in feature_names
        assert "so_rate" in feature_names
        assert "bb_rate" in feature_names
        assert "age" in feature_names
        assert "level_aaa" in feature_names
        assert "pct_at_aaa" in feature_names
        assert "total_pa" in feature_names
        # New Statcast features
        assert "xba" in feature_names
        assert "has_statcast" in feature_names


class TestBatterTargetStats:
    """Tests for BATTER_TARGET_STATS constant."""

    def test_target_stats_content(self) -> None:
        """Target stats should include expected stats."""
        expected = {"hr", "so", "bb", "singles", "doubles", "triples", "sb"}
        assert set(BATTER_TARGET_STATS) == expected
