"""Tests for the shared FeatureStore cache."""

from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats
from fantasy_baseball_manager.pipeline.statcast_data import (
    StatcastBatterStats,
    StatcastPitcherStats,
)


class FakeStatcastSource:
    def __init__(
        self,
        batter_stats: dict[int, list[StatcastBatterStats]],
        pitcher_stats: dict[int, list[StatcastPitcherStats]],
    ) -> None:
        self._batter = batter_stats
        self._pitcher = pitcher_stats
        self.batter_call_count = 0
        self.pitcher_call_count = 0

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        self.batter_call_count += 1
        return self._batter.get(year, [])

    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]:
        self.pitcher_call_count += 1
        return self._pitcher.get(year, [])


class FakeBattedBallSource:
    def __init__(self, stats: dict[int, list[PitcherBattedBallStats]]) -> None:
        self._stats = stats
        self.call_count = 0

    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]:
        self.call_count += 1
        return self._stats.get(year, [])


class FakeSkillDataSource:
    def __init__(
        self,
        batter_stats: dict[int, list[BatterSkillStats]] | None = None,
    ) -> None:
        self._batter = batter_stats or {}
        self.batter_call_count = 0

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]:
        self.batter_call_count += 1
        return self._batter.get(year, [])

    def pitcher_skill_stats(self, year: int) -> list:
        return []


BATTER_STATCAST_2023 = StatcastBatterStats(
    player_id="mlbam1",
    name="Batter One",
    year=2023,
    pa=500,
    barrel_rate=0.10,
    hard_hit_rate=0.40,
    xwoba=0.350,
    xba=0.270,
    xslg=0.450,
)

PITCHER_STATCAST_2023 = StatcastPitcherStats(
    player_id="mlbam2",
    name="Pitcher One",
    year=2023,
    pa=600,
    xba=0.230,
    xslg=0.380,
    xwoba=0.300,
    xera=3.50,
    barrel_rate=0.07,
    hard_hit_rate=0.32,
)

BATTED_BALL_2023 = PitcherBattedBallStats(
    player_id="fg1",
    name="Pitcher One",
    year=2023,
    pa=700,
    gb_pct=0.45,
    fb_pct=0.35,
    ld_pct=0.20,
    iffb_pct=0.10,
)

BATTER_SKILL_2023 = BatterSkillStats(
    player_id="fg2",
    name="Batter One",
    year=2023,
    pa=500,
    barrel_rate=0.10,
    hard_hit_rate=0.40,
    exit_velo_avg=89.0,
    exit_velo_max=110.0,
    chase_rate=0.28,
    whiff_rate=0.24,
    sprint_speed=27.5,
)


class TestBatterStatcastCache:
    def test_loads_once_per_year(self) -> None:
        source = FakeStatcastSource({2023: [BATTER_STATCAST_2023]}, {})
        store = FeatureStore(
            statcast_source=source,
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
        )
        store.batter_statcast(2023)
        store.batter_statcast(2023)
        assert source.batter_call_count == 1

    def test_keyed_by_player_id(self) -> None:
        source = FakeStatcastSource({2023: [BATTER_STATCAST_2023]}, {})
        store = FeatureStore(
            statcast_source=source,
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
        )
        result = store.batter_statcast(2023)
        assert "mlbam1" in result
        assert result["mlbam1"] is BATTER_STATCAST_2023

    def test_empty_source_returns_empty_dict(self) -> None:
        source = FakeStatcastSource({}, {})
        store = FeatureStore(
            statcast_source=source,
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
        )
        result = store.batter_statcast(2023)
        assert result == {}


class TestPitcherStatcastCache:
    def test_loads_once_per_year(self) -> None:
        source = FakeStatcastSource({}, {2023: [PITCHER_STATCAST_2023]})
        store = FeatureStore(
            statcast_source=source,
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
        )
        store.pitcher_statcast(2023)
        store.pitcher_statcast(2023)
        assert source.pitcher_call_count == 1

    def test_keyed_by_player_id(self) -> None:
        source = FakeStatcastSource({}, {2023: [PITCHER_STATCAST_2023]})
        store = FeatureStore(
            statcast_source=source,
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
        )
        result = store.pitcher_statcast(2023)
        assert "mlbam2" in result
        assert result["mlbam2"] is PITCHER_STATCAST_2023


class TestPitcherBattedBallCache:
    def test_loads_once_per_year(self) -> None:
        bb_source = FakeBattedBallSource({2023: [BATTED_BALL_2023]})
        store = FeatureStore(
            statcast_source=FakeStatcastSource({}, {}),
            batted_ball_source=bb_source,
            skill_data_source=FakeSkillDataSource(),
        )
        store.pitcher_batted_ball(2023)
        store.pitcher_batted_ball(2023)
        assert bb_source.call_count == 1

    def test_keyed_by_player_id(self) -> None:
        bb_source = FakeBattedBallSource({2023: [BATTED_BALL_2023]})
        store = FeatureStore(
            statcast_source=FakeStatcastSource({}, {}),
            batted_ball_source=bb_source,
            skill_data_source=FakeSkillDataSource(),
        )
        result = store.pitcher_batted_ball(2023)
        assert "fg1" in result
        assert result["fg1"] is BATTED_BALL_2023


class TestBatterSkillCache:
    def test_loads_once_per_year(self) -> None:
        skill_source = FakeSkillDataSource({2023: [BATTER_SKILL_2023]})
        store = FeatureStore(
            statcast_source=FakeStatcastSource({}, {}),
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=skill_source,
        )
        store.batter_skill(2023)
        store.batter_skill(2023)
        assert skill_source.batter_call_count == 1

    def test_keyed_by_player_id(self) -> None:
        skill_source = FakeSkillDataSource({2023: [BATTER_SKILL_2023]})
        store = FeatureStore(
            statcast_source=FakeStatcastSource({}, {}),
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=skill_source,
        )
        result = store.batter_skill(2023)
        assert "fg2" in result
        assert result["fg2"] is BATTER_SKILL_2023

    def test_empty_source_returns_empty_dict(self) -> None:
        skill_source = FakeSkillDataSource()
        store = FeatureStore(
            statcast_source=FakeStatcastSource({}, {}),
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=skill_source,
        )
        result = store.batter_skill(2023)
        assert result == {}


class TestMultiYearCaching:
    def test_different_years_cached_independently(self) -> None:
        batter_2022 = StatcastBatterStats(
            player_id="mlbam1",
            name="Batter One",
            year=2022,
            pa=400,
            barrel_rate=0.08,
            hard_hit_rate=0.38,
            xwoba=0.330,
            xba=0.260,
            xslg=0.420,
        )
        source = FakeStatcastSource(
            {2022: [batter_2022], 2023: [BATTER_STATCAST_2023]}, {}
        )
        store = FeatureStore(
            statcast_source=source,
            batted_ball_source=FakeBattedBallSource({}),
            skill_data_source=FakeSkillDataSource(),
        )

        result_2022 = store.batter_statcast(2022)
        result_2023 = store.batter_statcast(2023)

        assert source.batter_call_count == 2
        assert result_2022["mlbam1"].pa == 400
        assert result_2023["mlbam1"].pa == 500

        # Accessing again should not call source
        store.batter_statcast(2022)
        store.batter_statcast(2023)
        assert source.batter_call_count == 2
