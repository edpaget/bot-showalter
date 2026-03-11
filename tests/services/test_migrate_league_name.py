from __future__ import annotations

from fantasy_baseball_manager.domain import KeeperCost, LeagueKeeper
from fantasy_baseball_manager.services.keeper_migration import migrate_league_name
from tests.fakes.repos import FakeKeeperCostRepo, FakeLeagueKeeperRepo


class TestMigrateLeagueName:
    def test_renames_keeper_cost_rows(self) -> None:
        keeper_repo = FakeKeeperCostRepo(
            [
                KeeperCost(player_id=1, season=2025, league="keeper", cost=10.0, source="auction"),
                KeeperCost(player_id=2, season=2025, league="keeper", cost=20.0, source="auction"),
            ]
        )
        league_keeper_repo = FakeLeagueKeeperRepo()

        kc_count, lk_count = migrate_league_name(keeper_repo, league_keeper_repo, "keeper", "h2h")

        assert kc_count == 2
        assert lk_count == 0
        assert len(keeper_repo.find_by_season_league(2025, "h2h")) == 2
        assert len(keeper_repo.find_by_season_league(2025, "keeper")) == 0

    def test_renames_league_keeper_rows(self) -> None:
        keeper_repo = FakeKeeperCostRepo()
        league_keeper_repo = FakeLeagueKeeperRepo(
            [
                LeagueKeeper(player_id=1, season=2025, league="keeper", team_name="Team A"),
                LeagueKeeper(player_id=2, season=2025, league="keeper", team_name="Team B"),
                LeagueKeeper(player_id=3, season=2025, league="keeper", team_name="Team A"),
            ]
        )

        kc_count, lk_count = migrate_league_name(keeper_repo, league_keeper_repo, "keeper", "h2h")

        assert kc_count == 0
        assert lk_count == 3
        assert len(league_keeper_repo.find_by_season_league(2025, "h2h")) == 3
        assert len(league_keeper_repo.find_by_season_league(2025, "keeper")) == 0

    def test_returns_zero_when_no_matching_rows(self) -> None:
        keeper_repo = FakeKeeperCostRepo()
        league_keeper_repo = FakeLeagueKeeperRepo()

        kc_count, lk_count = migrate_league_name(keeper_repo, league_keeper_repo, "old", "new")

        assert kc_count == 0
        assert lk_count == 0

    def test_does_not_affect_other_leagues(self) -> None:
        keeper_repo = FakeKeeperCostRepo(
            [
                KeeperCost(player_id=1, season=2025, league="keeper", cost=10.0, source="auction"),
                KeeperCost(player_id=2, season=2025, league="redraft", cost=20.0, source="auction"),
            ]
        )
        league_keeper_repo = FakeLeagueKeeperRepo(
            [
                LeagueKeeper(player_id=3, season=2025, league="keeper", team_name="Team A"),
                LeagueKeeper(player_id=4, season=2025, league="dynasty", team_name="Team B"),
            ]
        )

        kc_count, lk_count = migrate_league_name(keeper_repo, league_keeper_repo, "keeper", "h2h")

        assert kc_count == 1
        assert lk_count == 1
        # Other leagues untouched
        assert len(keeper_repo.find_by_season_league(2025, "redraft")) == 1
        assert len(league_keeper_repo.find_by_season_league(2025, "dynasty")) == 1
