from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import LeagueKeeper
from fantasy_baseball_manager.repos.league_keeper_repo import SqliteLeagueKeeperRepo
from tests.helpers import seed_player


class TestSqliteLeagueKeeperRepo:
    def test_upsert_and_find_by_season_league(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        seed_player(conn, player_id=2, name_first="Shohei", name_last="Ohtani")
        repo = SqliteLeagueKeeperRepo(SingleConnectionProvider(conn))

        keepers = [
            LeagueKeeper(player_id=1, season=2026, league="dynasty", team_name="Team A", cost=25.0),
            LeagueKeeper(player_id=2, season=2026, league="dynasty", team_name="Team B", cost=30.0),
        ]
        count = repo.upsert_batch(keepers)
        conn.commit()

        assert count == 2
        result = repo.find_by_season_league(2026, "dynasty")
        assert len(result) == 2
        assert {k.player_id for k in result} == {1, 2}
        assert result[0].id is not None
        conn.close()

    def test_upsert_idempotent(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        repo = SqliteLeagueKeeperRepo(SingleConnectionProvider(conn))

        keeper = LeagueKeeper(player_id=1, season=2026, league="dynasty", team_name="Team A", cost=25.0)
        repo.upsert_batch([keeper])
        conn.commit()

        # Upsert again with different team/cost — should update, not duplicate
        updated = LeagueKeeper(player_id=1, season=2026, league="dynasty", team_name="Team B", cost=30.0)
        repo.upsert_batch([updated])
        conn.commit()

        result = repo.find_by_season_league(2026, "dynasty")
        assert len(result) == 1
        assert result[0].team_name == "Team B"
        assert result[0].cost == 30.0
        conn.close()

    def test_find_by_team(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        seed_player(conn, player_id=2, name_first="Shohei", name_last="Ohtani")
        seed_player(conn, player_id=3, name_first="Aaron", name_last="Judge")
        repo = SqliteLeagueKeeperRepo(SingleConnectionProvider(conn))

        keepers = [
            LeagueKeeper(player_id=1, season=2026, league="dynasty", team_name="Team A"),
            LeagueKeeper(player_id=2, season=2026, league="dynasty", team_name="Team A"),
            LeagueKeeper(player_id=3, season=2026, league="dynasty", team_name="Team B"),
        ]
        repo.upsert_batch(keepers)
        conn.commit()

        team_a = repo.find_by_team(2026, "dynasty", "Team A")
        assert len(team_a) == 2
        assert {k.player_id for k in team_a} == {1, 2}

        team_b = repo.find_by_team(2026, "dynasty", "Team B")
        assert len(team_b) == 1
        assert team_b[0].player_id == 3
        conn.close()

    def test_delete_by_season_league(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        seed_player(conn, player_id=2, name_first="Shohei", name_last="Ohtani")
        repo = SqliteLeagueKeeperRepo(SingleConnectionProvider(conn))

        keepers = [
            LeagueKeeper(player_id=1, season=2026, league="dynasty", team_name="Team A"),
            LeagueKeeper(player_id=2, season=2026, league="dynasty", team_name="Team B"),
        ]
        repo.upsert_batch(keepers)
        conn.commit()

        deleted = repo.delete_by_season_league(2026, "dynasty")
        conn.commit()

        assert deleted == 2
        result = repo.find_by_season_league(2026, "dynasty")
        assert len(result) == 0
        conn.close()

    def test_different_leagues_isolated(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        repo = SqliteLeagueKeeperRepo(SingleConnectionProvider(conn))

        repo.upsert_batch(
            [
                LeagueKeeper(player_id=1, season=2026, league="dynasty", team_name="Team A"),
            ]
        )
        conn.commit()

        # Different league should be empty
        result = repo.find_by_season_league(2026, "redraft")
        assert len(result) == 0

        # Original league still has the keeper
        result = repo.find_by_season_league(2026, "dynasty")
        assert len(result) == 1
        conn.close()

    def test_optional_fields(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        repo = SqliteLeagueKeeperRepo(SingleConnectionProvider(conn))

        # No cost or source
        keeper = LeagueKeeper(player_id=1, season=2026, league="dynasty", team_name="Team A")
        repo.upsert_batch([keeper])
        conn.commit()

        result = repo.find_by_season_league(2026, "dynasty")
        assert len(result) == 1
        assert result[0].cost is None
        assert result[0].source is None
        conn.close()
