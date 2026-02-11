"""Tests for PlayerStatProfile dataclass and feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from fantasy_baseball_manager.context import init_context, reset_context
from fantasy_baseball_manager.contextual.identity.stat_profile import (
    BATTER_STAT_ATTR_MAP,
    PITCHER_STAT_ATTR_MAP,
    PlayerStatProfile,
    PlayerStatProfileBuilder,
    _compute_raw_rates,
)
from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    PITCHER_TARGET_STATS,
)
from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
from fantasy_baseball_manager.result import Ok

# ---------------------------------------------------------------------------
# Helpers for building test data
# ---------------------------------------------------------------------------

def _batting_season(
    player_id: str = "p1",
    name: str = "Test Batter",
    year: int = 2023,
    age: int = 28,
    pa: int = 600,
    h: int = 150,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 25,
    bb: int = 60,
    so: int = 120,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=pa - bb - 5,
        h=h,
        singles=h - doubles - triples - hr,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=5,
        sf=5,
        sh=0,
        sb=10,
        cs=3,
        r=80,
        rbi=90,
    )


def _pitching_season(
    player_id: str = "p2",
    name: str = "Test Pitcher",
    year: int = 2023,
    age: int = 30,
    ip: float = 180.0,
    h: int = 150,
    bb: int = 50,
    so: int = 200,
    hr: int = 20,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        ip=ip,
        g=32,
        gs=32,
        er=70,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=5,
        w=12,
        sv=0,
        hld=0,
        bs=0,
    )


# ---------------------------------------------------------------------------
# Step 1: PlayerStatProfile dataclass + stat mapping constants
# ---------------------------------------------------------------------------

class TestStatMappingConstants:
    def test_batter_stat_attr_map_keys_match_target_stats(self) -> None:
        assert set(BATTER_STAT_ATTR_MAP.keys()) == set(BATTER_TARGET_STATS)

    def test_pitcher_stat_attr_map_keys_match_target_stats(self) -> None:
        assert set(PITCHER_STAT_ATTR_MAP.keys()) == set(PITCHER_TARGET_STATS)

    def test_batter_doubles_maps_to_doubles(self) -> None:
        assert BATTER_STAT_ATTR_MAP["2b"] == "doubles"

    def test_batter_triples_maps_to_triples(self) -> None:
        assert BATTER_STAT_ATTR_MAP["3b"] == "triples"

    def test_batter_direct_mappings(self) -> None:
        for stat in ("hr", "so", "bb", "h"):
            assert BATTER_STAT_ATTR_MAP[stat] == stat

    def test_pitcher_direct_mappings(self) -> None:
        for stat in ("so", "h", "bb", "hr"):
            assert PITCHER_STAT_ATTR_MAP[stat] == stat


class TestPlayerStatProfileCreation:
    def test_creation_with_all_fields(self) -> None:
        profile = PlayerStatProfile(
            player_id="p1",
            name="Test Player",
            year=2024,
            player_type="batter",
            age=28,
            handedness=None,
            rates_career={"hr": 0.04},
            rates_3yr={"hr": 0.05},
            rates_1yr={"hr": 0.06},
            rates_30d=None,
            opportunities_career=2000.0,
            opportunities_3yr=1800.0,
            opportunities_1yr=600.0,
        )
        assert profile.player_id == "p1"
        assert profile.year == 2024
        assert profile.player_type == "batter"

    def test_frozen_immutability(self) -> None:
        profile = PlayerStatProfile(
            player_id="p1",
            name="Test",
            year=2024,
            player_type="batter",
            age=28,
            handedness=None,
            rates_career={"hr": 0.04},
            rates_3yr=None,
            rates_1yr=None,
            rates_30d=None,
            opportunities_career=600.0,
            opportunities_3yr=None,
            opportunities_1yr=None,
        )
        with pytest.raises(AttributeError):
            profile.age = 29  # type: ignore[misc]

    def test_optional_fields_default_none(self) -> None:
        profile = PlayerStatProfile(
            player_id="p1",
            name="Test",
            year=2024,
            player_type="batter",
            age=28,
            handedness=None,
            rates_career={"hr": 0.04},
            rates_3yr=None,
            rates_1yr=None,
            rates_30d=None,
            opportunities_career=600.0,
            opportunities_3yr=None,
            opportunities_1yr=None,
        )
        assert profile.rates_3yr is None
        assert profile.rates_1yr is None
        assert profile.rates_30d is None
        assert profile.opportunities_3yr is None
        assert profile.opportunities_1yr is None


# ---------------------------------------------------------------------------
# Step 2: to_feature_vector() and feature_names()
# ---------------------------------------------------------------------------

class TestToFeatureVector:
    def _make_batter_profile(
        self,
        rates_career: dict[str, float] | None = None,
        rates_3yr: dict[str, float] | None = None,
        rates_1yr: dict[str, float] | None = None,
        age: int = 28,
    ) -> PlayerStatProfile:
        default_rates = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01}
        return PlayerStatProfile(
            player_id="p1",
            name="Test",
            year=2024,
            player_type="batter",
            age=age,
            handedness=None,
            rates_career=rates_career or default_rates,
            rates_3yr=rates_3yr,
            rates_1yr=rates_1yr,
            rates_30d=None,
            opportunities_career=2000.0,
            opportunities_3yr=1800.0 if rates_3yr else None,
            opportunities_1yr=600.0 if rates_1yr else None,
        )

    def _make_pitcher_profile(
        self,
        rates_career: dict[str, float] | None = None,
        rates_3yr: dict[str, float] | None = None,
        rates_1yr: dict[str, float] | None = None,
        age: int = 30,
    ) -> PlayerStatProfile:
        default_rates = {"so": 0.25, "h": 0.20, "bb": 0.08, "hr": 0.03}
        return PlayerStatProfile(
            player_id="p2",
            name="Test Pitcher",
            year=2024,
            player_type="pitcher",
            age=age,
            handedness=None,
            rates_career=rates_career or default_rates,
            rates_3yr=rates_3yr,
            rates_1yr=rates_1yr,
            rates_30d=None,
            opportunities_career=5000.0,
            opportunities_3yr=4000.0 if rates_3yr else None,
            opportunities_1yr=1500.0 if rates_1yr else None,
        )

    def test_batter_vector_length(self) -> None:
        profile = self._make_batter_profile(
            rates_3yr={"hr": 0.05, "so": 0.18, "bb": 0.11, "h": 0.26, "2b": 0.06, "3b": 0.01},
            rates_1yr={"hr": 0.06, "so": 0.15, "bb": 0.12, "h": 0.27, "2b": 0.07, "3b": 0.02},
        )
        vec = profile.to_feature_vector()
        assert len(vec) == 19  # 6*3 + 1 (age)

    def test_pitcher_vector_length(self) -> None:
        profile = self._make_pitcher_profile(
            rates_3yr={"so": 0.26, "h": 0.19, "bb": 0.07, "hr": 0.02},
            rates_1yr={"so": 0.27, "h": 0.18, "bb": 0.06, "hr": 0.01},
        )
        vec = profile.to_feature_vector()
        assert len(vec) == 13  # 4*3 + 1 (age)

    def test_batter_vector_ordering_career_3yr_1yr_age(self) -> None:
        career = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01}
        three_yr = {"hr": 0.05, "so": 0.18, "bb": 0.11, "h": 0.26, "2b": 0.06, "3b": 0.02}
        one_yr = {"hr": 0.06, "so": 0.15, "bb": 0.12, "h": 0.27, "2b": 0.07, "3b": 0.03}
        profile = self._make_batter_profile(
            rates_career=career,
            rates_3yr=three_yr,
            rates_1yr=one_yr,
            age=28,
        )
        vec = profile.to_feature_vector()
        # Last element is age
        assert vec[-1] == 28.0
        # Career rates come first in BATTER_TARGET_STATS order
        stat_order = list(BATTER_TARGET_STATS)
        for i, stat in enumerate(stat_order):
            assert vec[i] == pytest.approx(career[stat])
        for i, stat in enumerate(stat_order):
            assert vec[len(stat_order) + i] == pytest.approx(three_yr[stat])
        for i, stat in enumerate(stat_order):
            assert vec[2 * len(stat_order) + i] == pytest.approx(one_yr[stat])

    def test_fallback_when_3yr_is_none(self) -> None:
        career = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01}
        one_yr = {"hr": 0.06, "so": 0.15, "bb": 0.12, "h": 0.27, "2b": 0.07, "3b": 0.03}
        profile = self._make_batter_profile(
            rates_career=career,
            rates_3yr=None,
            rates_1yr=one_yr,
        )
        vec = profile.to_feature_vector()
        n = len(BATTER_TARGET_STATS)
        # 3yr slot should fall back to career
        for i, stat in enumerate(list(BATTER_TARGET_STATS)):
            assert vec[n + i] == pytest.approx(career[stat])
        # 1yr slot is populated
        for i, stat in enumerate(list(BATTER_TARGET_STATS)):
            assert vec[2 * n + i] == pytest.approx(one_yr[stat])

    def test_fallback_when_both_3yr_and_1yr_are_none(self) -> None:
        career = {"hr": 0.04, "so": 0.20, "bb": 0.10, "h": 0.25, "2b": 0.05, "3b": 0.01}
        profile = self._make_batter_profile(rates_career=career)
        vec = profile.to_feature_vector()
        n = len(BATTER_TARGET_STATS)
        # All three horizons should be career rates
        for i, stat in enumerate(list(BATTER_TARGET_STATS)):
            assert vec[i] == pytest.approx(career[stat])
            assert vec[n + i] == pytest.approx(career[stat])
            assert vec[2 * n + i] == pytest.approx(career[stat])

    def test_feature_names_matches_vector_length_batter(self) -> None:
        names = PlayerStatProfile.feature_names("batter")
        profile = self._make_batter_profile(
            rates_3yr={"hr": 0.05, "so": 0.18, "bb": 0.11, "h": 0.26, "2b": 0.06, "3b": 0.01},
            rates_1yr={"hr": 0.06, "so": 0.15, "bb": 0.12, "h": 0.27, "2b": 0.07, "3b": 0.02},
        )
        vec = profile.to_feature_vector()
        assert len(names) == len(vec) == 19

    def test_feature_names_matches_vector_length_pitcher(self) -> None:
        names = PlayerStatProfile.feature_names("pitcher")
        profile = self._make_pitcher_profile(
            rates_3yr={"so": 0.26, "h": 0.19, "bb": 0.07, "hr": 0.02},
            rates_1yr={"so": 0.27, "h": 0.18, "bb": 0.06, "hr": 0.01},
        )
        vec = profile.to_feature_vector()
        assert len(names) == len(vec) == 13

    def test_feature_names_contain_horizon_prefix(self) -> None:
        names = PlayerStatProfile.feature_names("batter")
        assert names[0].startswith("career_")
        n = len(BATTER_TARGET_STATS)
        assert names[n].startswith("3yr_")
        assert names[2 * n].startswith("1yr_")
        assert names[-1] == "age"

    def test_returns_numpy_array(self) -> None:
        profile = self._make_batter_profile()
        vec = profile.to_feature_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float64


# ---------------------------------------------------------------------------
# Step 3: _compute_raw_rates() helper
# ---------------------------------------------------------------------------

class TestComputeRawRates:
    def test_single_batting_season_rate(self) -> None:
        season = _batting_season(pa=600, hr=30, so=120, bb=60, h=150, doubles=30, triples=5)
        result = _compute_raw_rates([season], "batter")
        assert result is not None
        rates, opps = result
        assert opps == 600.0
        assert rates["hr"] == pytest.approx(30 / 600)
        assert rates["so"] == pytest.approx(120 / 600)
        assert rates["bb"] == pytest.approx(60 / 600)
        assert rates["h"] == pytest.approx(150 / 600)
        assert rates["2b"] == pytest.approx(30 / 600)
        assert rates["3b"] == pytest.approx(5 / 600)

    def test_multi_season_aggregation(self) -> None:
        s1 = _batting_season(pa=500, hr=20, so=100, bb=50, h=130, doubles=25, triples=3)
        s2 = _batting_season(pa=400, hr=10, so=80, bb=40, h=100, doubles=20, triples=2)
        result = _compute_raw_rates([s1, s2], "batter")
        assert result is not None
        rates, opps = result
        assert opps == 900.0
        assert rates["hr"] == pytest.approx(30 / 900)
        assert rates["h"] == pytest.approx(230 / 900)

    def test_single_pitching_season(self) -> None:
        season = _pitching_season(ip=180.0, so=200, h=150, bb=50, hr=20)
        result = _compute_raw_rates([season], "pitcher")
        assert result is not None
        rates, opps = result
        # outs = ip * 3
        assert opps == 540.0
        assert rates["so"] == pytest.approx(200 / 540)
        assert rates["h"] == pytest.approx(150 / 540)
        assert rates["bb"] == pytest.approx(50 / 540)
        assert rates["hr"] == pytest.approx(20 / 540)

    def test_zero_opportunities_returns_zero_rates(self) -> None:
        season = _batting_season(pa=0, hr=0, so=0, bb=0, h=0, doubles=0, triples=0)
        result = _compute_raw_rates([season], "batter")
        assert result is not None
        rates, opps = result
        assert opps == 0.0
        for val in rates.values():
            assert val == 0.0

    def test_empty_list_returns_none(self) -> None:
        result = _compute_raw_rates([], "batter")
        assert result is None


# ---------------------------------------------------------------------------
# Step 4: PlayerStatProfileBuilder.build_profile()
# ---------------------------------------------------------------------------

class TestBuildProfile:
    def test_three_seasons_all_horizons_populated(self) -> None:
        seasons = {
            2021: _batting_season(year=2021, age=25, pa=500, hr=20),
            2022: _batting_season(year=2022, age=26, pa=550, hr=25),
            2023: _batting_season(year=2023, age=27, pa=600, hr=30),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p1", name="Test", seasons=seasons, year=2024, player_type="batter"
        )
        assert profile.rates_career is not None
        assert profile.rates_3yr is not None
        assert profile.rates_1yr is not None
        assert profile.year == 2024
        assert profile.age == 28  # 27 + 1

    def test_one_season_only_3yr_is_none(self) -> None:
        seasons = {
            2023: _batting_season(year=2023, age=27, pa=600, hr=30),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p1", name="Test", seasons=seasons, year=2024, player_type="batter"
        )
        assert profile.rates_career is not None
        assert profile.rates_3yr is None  # Need >=2 seasons in window
        assert profile.rates_1yr is not None

    def test_no_recent_year_1yr_is_none(self) -> None:
        seasons = {
            2020: _batting_season(year=2020, age=24, pa=500, hr=20),
            2021: _batting_season(year=2021, age=25, pa=550, hr=25),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p1", name="Test", seasons=seasons, year=2024, player_type="batter"
        )
        assert profile.rates_career is not None
        assert profile.rates_3yr is None  # 2020, 2021 not in 2021-2023 range
        assert profile.rates_1yr is None  # No 2023 data

    def test_career_rate_aggregation_math(self) -> None:
        seasons = {
            2022: _batting_season(year=2022, age=26, pa=500, hr=20),
            2023: _batting_season(year=2023, age=27, pa=600, hr=30),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p1", name="Test", seasons=seasons, year=2024, player_type="batter"
        )
        assert profile.rates_career["hr"] == pytest.approx(50 / 1100)
        assert profile.opportunities_career == 1100.0

    def test_3yr_window_uses_correct_years(self) -> None:
        # year=2024, so 3yr = 2021, 2022, 2023
        seasons = {
            2020: _batting_season(year=2020, age=24, pa=400, hr=10),
            2021: _batting_season(year=2021, age=25, pa=500, hr=20),
            2022: _batting_season(year=2022, age=26, pa=550, hr=25),
            2023: _batting_season(year=2023, age=27, pa=600, hr=30),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p1", name="Test", seasons=seasons, year=2024, player_type="batter"
        )
        # 3yr should include 2021, 2022, 2023 only
        assert profile.rates_3yr is not None
        assert profile.rates_3yr["hr"] == pytest.approx(75 / 1650)
        assert profile.opportunities_3yr == 1650.0

    def test_1yr_uses_only_prior_year(self) -> None:
        seasons = {
            2022: _batting_season(year=2022, age=26, pa=500, hr=20),
            2023: _batting_season(year=2023, age=27, pa=600, hr=30),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p1", name="Test", seasons=seasons, year=2024, player_type="batter"
        )
        assert profile.rates_1yr is not None
        assert profile.rates_1yr["hr"] == pytest.approx(30 / 600)
        assert profile.opportunities_1yr == 600.0

    def test_pitcher_denominators(self) -> None:
        seasons = {
            2023: _pitching_season(year=2023, age=29, ip=180.0, so=200, h=150, bb=50, hr=20),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p2", name="Pitcher", seasons=seasons, year=2024, player_type="pitcher"
        )
        assert profile.opportunities_career == 540.0  # 180 * 3
        assert profile.rates_career["so"] == pytest.approx(200 / 540)

    def test_age_projection(self) -> None:
        seasons = {
            2023: _batting_season(year=2023, age=27, pa=600, hr=30),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p1", name="Test", seasons=seasons, year=2024, player_type="batter"
        )
        assert profile.age == 28  # 27 + (2024 - 2023)

    def test_age_projection_multi_year_gap(self) -> None:
        seasons = {
            2021: _batting_season(year=2021, age=25, pa=500, hr=20),
        }
        builder = PlayerStatProfileBuilder()
        profile = builder.build_profile(
            player_id="p1", name="Test", seasons=seasons, year=2024, player_type="batter"
        )
        # Most recent season is 2021, age=25. Project to 2024: 25 + (2024 - 2021) = 28
        assert profile.age == 28


# ---------------------------------------------------------------------------
# Step 5: PlayerStatProfileBuilder.build_all_profiles()
# ---------------------------------------------------------------------------

class _FakeBattingSource:
    """Fake data source returning batting stats keyed by context year."""

    def __init__(self, data_by_year: dict[int, list[BattingSeasonStats]]) -> None:
        self._data = data_by_year

    def __call__(self, query: object) -> Ok[list[BattingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        year = get_context().year
        return Ok(self._data.get(year, []))


class _FakePitchingSource:
    """Fake data source returning pitching stats keyed by context year."""

    def __init__(self, data_by_year: dict[int, list[PitchingSeasonStats]]) -> None:
        self._data = data_by_year

    def __call__(self, query: object) -> Ok[list[PitchingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        year = get_context().year
        return Ok(self._data.get(year, []))


class TestBuildAllProfiles:
    def setup_method(self) -> None:
        init_context(year=2024)

    def teardown_method(self) -> None:
        reset_context()

    def test_builds_profiles_for_all_players(self) -> None:
        batting_data = {
            2022: [_batting_season(player_id="b1", name="Batter1", year=2022, age=26, pa=500)],
            2023: [
                _batting_season(player_id="b1", name="Batter1", year=2023, age=27, pa=600),
                _batting_season(player_id="b2", name="Batter2", year=2023, age=25, pa=550),
            ],
        }
        pitching_data = {
            2023: [_pitching_season(player_id="p1", name="Pitcher1", year=2023, age=30, ip=180.0)],
        }
        builder = PlayerStatProfileBuilder()
        profiles = builder.build_all_profiles(
            batting_source=_FakeBattingSource(batting_data),
            pitching_source=_FakePitchingSource(pitching_data),
            year=2024,
            history_years=[2022, 2023],
        )
        ids = {p.player_id for p in profiles}
        assert "b1" in ids
        assert "b2" in ids
        assert "p1" in ids
        assert len(profiles) == 3

    def test_old_data_players_still_get_profiles(self) -> None:
        batting_data = {
            2019: [_batting_season(player_id="old", name="OldPlayer", year=2019, age=30, pa=500)],
        }
        builder = PlayerStatProfileBuilder()
        profiles = builder.build_all_profiles(
            batting_source=_FakeBattingSource(batting_data),
            pitching_source=_FakePitchingSource({}),
            year=2024,
            history_years=[2019, 2020, 2021, 2022, 2023],
        )
        assert len(profiles) == 1
        assert profiles[0].player_id == "old"

    def test_min_opportunities_filter(self) -> None:
        batting_data = {
            2023: [
                _batting_season(player_id="big", name="Big", year=2023, pa=600),
                _batting_season(player_id="small", name="Small", year=2023, pa=30),
            ],
        }
        builder = PlayerStatProfileBuilder()
        profiles = builder.build_all_profiles(
            batting_source=_FakeBattingSource(batting_data),
            pitching_source=_FakePitchingSource({}),
            year=2024,
            history_years=[2023],
            min_opportunities=50.0,
        )
        ids = {p.player_id for p in profiles}
        assert "big" in ids
        assert "small" not in ids

    def test_uses_new_context_for_each_year(self) -> None:
        """Verifies data is fetched per-year by checking the source gets called
        with different context years."""
        call_years: list[int] = []

        class _TrackingSource:
            def __call__(self, query: object) -> Ok[list[BattingSeasonStats]]:
                from fantasy_baseball_manager.context import get_context

                call_years.append(get_context().year)
                return Ok([])

        builder = PlayerStatProfileBuilder()
        builder.build_all_profiles(
            batting_source=_TrackingSource(),
            pitching_source=_FakePitchingSource({}),
            year=2024,
            history_years=[2021, 2022, 2023],
        )
        assert call_years == [2021, 2022, 2023]
