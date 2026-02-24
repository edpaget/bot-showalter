from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.services.player_universe import (
    PlayerUniverseProvider,
    StatsBasedPlayerUniverse,
)
from tests.fakes.repos import FakeBattingStatsRepo, FakePitchingStatsRepo


class TestStatsBasedPlayerUniverseBatters:
    def test_returns_player_ids_from_prior_season(self) -> None:
        batting_repo = FakeBattingStatsRepo(
            stats=[
                BattingStats(player_id=1, season=2025, source="fangraphs", pa=500),
                BattingStats(player_id=2, season=2025, source="fangraphs", pa=400),
            ]
        )
        pitching_repo = FakePitchingStatsRepo()
        universe = StatsBasedPlayerUniverse(batting_repo=batting_repo, pitching_repo=pitching_repo)

        result = universe.get_player_ids(2026, "batter")

        assert result == {1, 2}

    def test_source_filtering(self) -> None:
        batting_repo = FakeBattingStatsRepo(
            stats=[
                BattingStats(player_id=1, season=2025, source="fangraphs", pa=500),
                BattingStats(player_id=2, season=2025, source="bbref", pa=400),
                BattingStats(player_id=3, season=2025, source="fangraphs", pa=300),
            ]
        )
        pitching_repo = FakePitchingStatsRepo()
        universe = StatsBasedPlayerUniverse(batting_repo=batting_repo, pitching_repo=pitching_repo)

        result = universe.get_player_ids(2026, "batter", source="fangraphs")

        assert result == {1, 3}

    def test_min_pa_filtering(self) -> None:
        batting_repo = FakeBattingStatsRepo(
            stats=[
                BattingStats(player_id=1, season=2025, source="fangraphs", pa=500),
                BattingStats(player_id=2, season=2025, source="fangraphs", pa=50),
                BattingStats(player_id=3, season=2025, source="fangraphs", pa=100),
            ]
        )
        pitching_repo = FakePitchingStatsRepo()
        universe = StatsBasedPlayerUniverse(batting_repo=batting_repo, pitching_repo=pitching_repo)

        result = universe.get_player_ids(2026, "batter", min_pa=100)

        assert result == {1, 3}

    def test_empty_prior_season_returns_empty_set(self) -> None:
        batting_repo = FakeBattingStatsRepo()
        pitching_repo = FakePitchingStatsRepo()
        universe = StatsBasedPlayerUniverse(batting_repo=batting_repo, pitching_repo=pitching_repo)

        result = universe.get_player_ids(2026, "batter")

        assert result == set()

    def test_uses_season_minus_one_not_season_itself(self) -> None:
        batting_repo = FakeBattingStatsRepo(
            stats=[
                BattingStats(player_id=1, season=2026, source="fangraphs", pa=500),
            ]
        )
        pitching_repo = FakePitchingStatsRepo()
        universe = StatsBasedPlayerUniverse(batting_repo=batting_repo, pitching_repo=pitching_repo)

        result = universe.get_player_ids(2026, "batter")

        assert result == set()


class TestStatsBasedPlayerUniversePitchers:
    def test_returns_player_ids_from_prior_season(self) -> None:
        batting_repo = FakeBattingStatsRepo()
        pitching_repo = FakePitchingStatsRepo(
            stats=[
                PitchingStats(player_id=10, season=2025, source="fangraphs", ip=180.0),
                PitchingStats(player_id=11, season=2025, source="fangraphs", ip=60.0),
            ]
        )
        universe = StatsBasedPlayerUniverse(batting_repo=batting_repo, pitching_repo=pitching_repo)

        result = universe.get_player_ids(2026, "pitcher")

        assert result == {10, 11}

    def test_min_ip_filtering(self) -> None:
        batting_repo = FakeBattingStatsRepo()
        pitching_repo = FakePitchingStatsRepo(
            stats=[
                PitchingStats(player_id=10, season=2025, source="fangraphs", ip=180.0),
                PitchingStats(player_id=11, season=2025, source="fangraphs", ip=10.0),
                PitchingStats(player_id=12, season=2025, source="fangraphs", ip=50.0),
            ]
        )
        universe = StatsBasedPlayerUniverse(batting_repo=batting_repo, pitching_repo=pitching_repo)

        result = universe.get_player_ids(2026, "pitcher", min_ip=50.0)

        assert result == {10, 12}


class TestPlayerUniverseProviderProtocol:
    def test_satisfies_protocol(self) -> None:
        batting_repo = FakeBattingStatsRepo()
        pitching_repo = FakePitchingStatsRepo()
        universe = StatsBasedPlayerUniverse(batting_repo=batting_repo, pitching_repo=pitching_repo)

        assert isinstance(universe, PlayerUniverseProvider)
