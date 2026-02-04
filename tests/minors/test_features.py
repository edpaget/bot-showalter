"""Tests for minors/features.py"""

from __future__ import annotations

import numpy as np
import pytest

from fantasy_baseball_manager.minors.features import (
    TYPICAL_AGE_BY_LEVEL,
    MLEBatterFeatureExtractor,
)
from fantasy_baseball_manager.minors.training_data import AggregatedMiLBStats
from fantasy_baseball_manager.minors.types import (
    MiLBStatcastStats,
    MinorLeagueLevel,
)


def _make_aggregated_stats(
    player_id: str = "12345",
    name: str = "Test Player",
    season: int = 2023,
    age: int = 23,
    total_pa: int = 500,
    highest_level: MinorLeagueLevel = MinorLeagueLevel.AAA,
    pct_at_aaa: float = 1.0,
    pct_at_aa: float = 0.0,
    pct_at_high_a: float = 0.0,
    pct_at_single_a: float = 0.0,
    hr_rate: float = 0.04,
    so_rate: float = 0.20,
    bb_rate: float = 0.10,
    hit_rate: float = 0.26,
    singles_rate: float = 0.16,
    doubles_rate: float = 0.05,
    triples_rate: float = 0.01,
    sb_rate: float = 0.10,
    iso: float = 0.180,
    avg: float = 0.280,
    obp: float = 0.360,
    slg: float = 0.460,
) -> AggregatedMiLBStats:
    """Create an AggregatedMiLBStats with sensible defaults."""
    return AggregatedMiLBStats(
        player_id=player_id,
        name=name,
        season=season,
        age=age,
        total_pa=total_pa,
        highest_level=highest_level,
        pct_at_aaa=pct_at_aaa,
        pct_at_aa=pct_at_aa,
        pct_at_high_a=pct_at_high_a,
        pct_at_single_a=pct_at_single_a,
        hr_rate=hr_rate,
        so_rate=so_rate,
        bb_rate=bb_rate,
        hit_rate=hit_rate,
        singles_rate=singles_rate,
        doubles_rate=doubles_rate,
        triples_rate=triples_rate,
        sb_rate=sb_rate,
        iso=iso,
        avg=avg,
        obp=obp,
        slg=slg,
    )


def _make_statcast_stats(
    player_id: str = "12345",
    season: int = 2023,
    pa: int = 500,
    xba: float = 0.270,
    xslg: float = 0.450,
    xwoba: float = 0.340,
    barrel_rate: float = 0.08,
    hard_hit_rate: float = 0.40,
    sprint_speed: float | None = 27.5,
) -> MiLBStatcastStats:
    """Create a MiLBStatcastStats with sensible defaults."""
    return MiLBStatcastStats(
        player_id=player_id,
        season=season,
        pa=pa,
        xba=xba,
        xslg=xslg,
        xwoba=xwoba,
        barrel_rate=barrel_rate,
        hard_hit_rate=hard_hit_rate,
        sprint_speed=sprint_speed,
    )


class TestMLEBatterFeatureExtractor:
    """Tests for MLEBatterFeatureExtractor."""

    def test_feature_names_returns_list(self) -> None:
        """feature_names() should return a list of strings."""
        extractor = MLEBatterFeatureExtractor()
        names = extractor.feature_names()

        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) == 32  # 25 base + 7 Statcast

    def test_n_features(self) -> None:
        """n_features() should return correct count."""
        extractor = MLEBatterFeatureExtractor()
        assert extractor.n_features() == 32

    def test_extract_basic_features(self) -> None:
        """Extract should return correct basic features."""
        stats = _make_aggregated_stats(
            hr_rate=0.04,
            so_rate=0.20,
            bb_rate=0.10,
            age=23,
            total_pa=500,
        )
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats)

        assert features is not None
        assert features.dtype == np.float32
        assert len(features) == 32

        # Check specific feature values
        names = extractor.feature_names()
        assert features[names.index("hr_rate")] == pytest.approx(0.04)
        assert features[names.index("so_rate")] == pytest.approx(0.20)
        assert features[names.index("bb_rate")] == pytest.approx(0.10)
        assert features[names.index("age")] == pytest.approx(23.0)
        assert features[names.index("total_pa")] == pytest.approx(500.0)

    def test_extract_age_features(self) -> None:
        """Age features should include age, age_squared, and age_for_level."""
        stats = _make_aggregated_stats(age=23, highest_level=MinorLeagueLevel.AAA)
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats)
        names = extractor.feature_names()

        assert features is not None
        assert features[names.index("age")] == 23.0
        assert features[names.index("age_squared")] == pytest.approx(23**2 / 1000.0)
        # AAA typical age is 25, so age_for_level = 23 - 25 = -2
        assert features[names.index("age_for_level")] == pytest.approx(-2.0)

    def test_extract_age_for_level_young_player(self) -> None:
        """Young player at AA should have negative age_for_level."""
        stats = _make_aggregated_stats(age=21, highest_level=MinorLeagueLevel.AA)
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats)
        names = extractor.feature_names()

        # AA typical age is 23, so age_for_level = 21 - 23 = -2
        assert features is not None
        assert features[names.index("age_for_level")] == pytest.approx(-2.0)

    def test_extract_age_for_level_old_player(self) -> None:
        """Old player at AA should have positive age_for_level."""
        stats = _make_aggregated_stats(age=26, highest_level=MinorLeagueLevel.AA)
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats)
        names = extractor.feature_names()

        # AA typical age is 23, so age_for_level = 26 - 23 = 3
        assert features is not None
        assert features[names.index("age_for_level")] == pytest.approx(3.0)

    def test_extract_level_one_hot(self) -> None:
        """Level should be one-hot encoded."""
        stats_aaa = _make_aggregated_stats(highest_level=MinorLeagueLevel.AAA)
        stats_aa = _make_aggregated_stats(highest_level=MinorLeagueLevel.AA)
        stats_high_a = _make_aggregated_stats(highest_level=MinorLeagueLevel.HIGH_A)
        stats_single_a = _make_aggregated_stats(highest_level=MinorLeagueLevel.SINGLE_A)

        extractor = MLEBatterFeatureExtractor()
        names = extractor.feature_names()

        features_aaa = extractor.extract(stats_aaa)
        assert features_aaa is not None
        assert features_aaa[names.index("level_aaa")] == 1.0
        assert features_aaa[names.index("level_aa")] == 0.0
        assert features_aaa[names.index("level_high_a")] == 0.0
        assert features_aaa[names.index("level_single_a")] == 0.0

        features_aa = extractor.extract(stats_aa)
        assert features_aa is not None
        assert features_aa[names.index("level_aaa")] == 0.0
        assert features_aa[names.index("level_aa")] == 1.0

        features_high_a = extractor.extract(stats_high_a)
        assert features_high_a is not None
        assert features_high_a[names.index("level_high_a")] == 1.0

        features_single_a = extractor.extract(stats_single_a)
        assert features_single_a is not None
        assert features_single_a[names.index("level_single_a")] == 1.0

    def test_extract_level_distribution(self) -> None:
        """Level distribution should reflect PA percentages."""
        stats = _make_aggregated_stats(
            pct_at_aaa=0.4,
            pct_at_aa=0.3,
            pct_at_high_a=0.2,
            pct_at_single_a=0.1,
        )
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats)
        names = extractor.feature_names()

        assert features is not None
        assert features[names.index("pct_at_aaa")] == pytest.approx(0.4)
        assert features[names.index("pct_at_aa")] == pytest.approx(0.3)
        assert features[names.index("pct_at_high_a")] == pytest.approx(0.2)
        assert features[names.index("pct_at_single_a")] == pytest.approx(0.1)

    def test_extract_sample_size_features(self) -> None:
        """Sample size features should be extracted correctly."""
        stats = _make_aggregated_stats(total_pa=500)
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats)
        names = extractor.feature_names()

        assert features is not None
        assert features[names.index("total_pa")] == pytest.approx(500.0)
        assert features[names.index("log_pa")] == pytest.approx(np.log(501))

    def test_extract_without_statcast(self) -> None:
        """Without Statcast, features should be zero-imputed."""
        stats = _make_aggregated_stats()
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats, statcast=None)
        names = extractor.feature_names()

        assert features is not None
        assert features[names.index("xba")] == 0.0
        assert features[names.index("xslg")] == 0.0
        assert features[names.index("xwoba")] == 0.0
        assert features[names.index("barrel_rate")] == 0.0
        assert features[names.index("hard_hit_rate")] == 0.0
        assert features[names.index("sprint_speed")] == 0.0
        assert features[names.index("has_statcast")] == 0.0

    def test_extract_with_statcast(self) -> None:
        """With Statcast, features should be populated."""
        stats = _make_aggregated_stats()
        statcast = _make_statcast_stats(
            xba=0.280,
            xslg=0.460,
            xwoba=0.350,
            barrel_rate=0.10,
            hard_hit_rate=0.45,
            sprint_speed=28.0,
        )
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats, statcast=statcast)
        names = extractor.feature_names()

        assert features is not None
        assert features[names.index("xba")] == pytest.approx(0.280)
        assert features[names.index("xslg")] == pytest.approx(0.460)
        assert features[names.index("xwoba")] == pytest.approx(0.350)
        assert features[names.index("barrel_rate")] == pytest.approx(0.10)
        assert features[names.index("hard_hit_rate")] == pytest.approx(0.45)
        assert features[names.index("sprint_speed")] == pytest.approx(28.0)
        assert features[names.index("has_statcast")] == 1.0

    def test_extract_with_statcast_missing_sprint_speed(self) -> None:
        """Missing sprint_speed should be zero-imputed."""
        stats = _make_aggregated_stats()
        statcast = _make_statcast_stats(sprint_speed=None)
        extractor = MLEBatterFeatureExtractor()

        features = extractor.extract(stats, statcast=statcast)
        names = extractor.feature_names()

        assert features is not None
        assert features[names.index("sprint_speed")] == 0.0
        assert features[names.index("has_statcast")] == 1.0  # Still has other Statcast

    def test_extract_below_min_pa_returns_none(self) -> None:
        """Below minimum PA should return None."""
        stats = _make_aggregated_stats(total_pa=40)
        extractor = MLEBatterFeatureExtractor(min_pa=50)

        assert extractor.extract(stats) is None

    def test_extract_at_min_pa_returns_features(self) -> None:
        """At minimum PA should return features."""
        stats = _make_aggregated_stats(total_pa=50)
        extractor = MLEBatterFeatureExtractor(min_pa=50)

        features = extractor.extract(stats)
        assert features is not None

    def test_extract_custom_min_pa(self) -> None:
        """Custom min_pa should be respected."""
        stats = _make_aggregated_stats(total_pa=100)

        extractor_100 = MLEBatterFeatureExtractor(min_pa=100)
        extractor_200 = MLEBatterFeatureExtractor(min_pa=200)

        assert extractor_100.extract(stats) is not None
        assert extractor_200.extract(stats) is None


class TestMLEBatterFeatureExtractorBatch:
    """Tests for extract_batch method."""

    def test_extract_batch_all_valid(self) -> None:
        """Batch extraction with all valid samples."""
        stats_list = [
            _make_aggregated_stats(player_id="1", total_pa=500),
            _make_aggregated_stats(player_id="2", total_pa=400),
            _make_aggregated_stats(player_id="3", total_pa=300),
        ]
        extractor = MLEBatterFeatureExtractor()

        features, indices = extractor.extract_batch(stats_list)

        assert features.shape == (3, 32)
        assert indices == [0, 1, 2]

    def test_extract_batch_some_invalid(self) -> None:
        """Batch extraction should skip invalid samples."""
        stats_list = [
            _make_aggregated_stats(player_id="1", total_pa=500),
            _make_aggregated_stats(player_id="2", total_pa=30),  # Below min_pa
            _make_aggregated_stats(player_id="3", total_pa=300),
        ]
        extractor = MLEBatterFeatureExtractor(min_pa=50)

        features, indices = extractor.extract_batch(stats_list)

        assert features.shape == (2, 32)
        assert indices == [0, 2]

    def test_extract_batch_all_invalid(self) -> None:
        """Batch extraction with all invalid samples."""
        stats_list = [
            _make_aggregated_stats(player_id="1", total_pa=30),
            _make_aggregated_stats(player_id="2", total_pa=40),
        ]
        extractor = MLEBatterFeatureExtractor(min_pa=50)

        features, indices = extractor.extract_batch(stats_list)

        assert features.shape == (0, 32)
        assert indices == []

    def test_extract_batch_empty_list(self) -> None:
        """Batch extraction with empty list."""
        extractor = MLEBatterFeatureExtractor()

        features, indices = extractor.extract_batch([])

        assert features.shape == (0, 32)
        assert indices == []

    def test_extract_batch_with_statcast_lookup(self) -> None:
        """Batch extraction with Statcast lookup."""
        stats_list = [
            _make_aggregated_stats(player_id="1", total_pa=500),
            _make_aggregated_stats(player_id="2", total_pa=400),
            _make_aggregated_stats(player_id="3", total_pa=300),
        ]
        statcast_lookup = {
            "1": _make_statcast_stats(player_id="1", xba=0.290),
            "3": _make_statcast_stats(player_id="3", xba=0.260),
            # Player 2 has no Statcast data
        }
        extractor = MLEBatterFeatureExtractor()

        features, indices = extractor.extract_batch(stats_list, statcast_lookup)
        names = extractor.feature_names()

        assert features.shape == (3, 32)
        assert indices == [0, 1, 2]

        # Player 1 has Statcast
        assert features[0, names.index("xba")] == pytest.approx(0.290)
        assert features[0, names.index("has_statcast")] == 1.0

        # Player 2 has no Statcast (zero-imputed)
        assert features[1, names.index("xba")] == 0.0
        assert features[1, names.index("has_statcast")] == 0.0

        # Player 3 has Statcast
        assert features[2, names.index("xba")] == pytest.approx(0.260)
        assert features[2, names.index("has_statcast")] == 1.0


class TestTypicalAgeByLevel:
    """Tests for TYPICAL_AGE_BY_LEVEL constant."""

    def test_typical_ages_defined(self) -> None:
        """All levels should have typical ages defined."""
        assert MinorLeagueLevel.AAA in TYPICAL_AGE_BY_LEVEL
        assert MinorLeagueLevel.AA in TYPICAL_AGE_BY_LEVEL
        assert MinorLeagueLevel.HIGH_A in TYPICAL_AGE_BY_LEVEL
        assert MinorLeagueLevel.SINGLE_A in TYPICAL_AGE_BY_LEVEL
        assert MinorLeagueLevel.ROOKIE in TYPICAL_AGE_BY_LEVEL

    def test_typical_ages_reasonable(self) -> None:
        """Typical ages should be reasonable values."""
        for level, age in TYPICAL_AGE_BY_LEVEL.items():
            assert 18 <= age <= 30, f"Unreasonable age {age} for {level}"

    def test_typical_ages_decrease_with_level(self) -> None:
        """Higher levels should have higher typical ages."""
        assert TYPICAL_AGE_BY_LEVEL[MinorLeagueLevel.AAA] > TYPICAL_AGE_BY_LEVEL[MinorLeagueLevel.AA]
        assert TYPICAL_AGE_BY_LEVEL[MinorLeagueLevel.AA] > TYPICAL_AGE_BY_LEVEL[MinorLeagueLevel.HIGH_A]
        assert TYPICAL_AGE_BY_LEVEL[MinorLeagueLevel.HIGH_A] > TYPICAL_AGE_BY_LEVEL[MinorLeagueLevel.SINGLE_A]
        assert TYPICAL_AGE_BY_LEVEL[MinorLeagueLevel.SINGLE_A] > TYPICAL_AGE_BY_LEVEL[MinorLeagueLevel.ROOKIE]
