from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.ingest.league_keeper_mapper import import_league_keepers
from fantasy_baseball_manager.repos.league_keeper_repo import SqliteLeagueKeeperRepo
from tests.helpers import seed_player


class TestImportLeagueKeepers:
    def _setup(self) -> tuple:
        conn = create_connection(":memory:")
        p1 = seed_player(conn, name_first="Mike", name_last="Trout")
        p2 = seed_player(conn, name_first="Shohei", name_last="Ohtani")
        repo = SqliteLeagueKeeperRepo(SingleConnectionProvider(conn))
        players = [
            Player(id=p1, name_first="Mike", name_last="Trout"),
            Player(id=p2, name_first="Shohei", name_last="Ohtani"),
        ]
        return conn, repo, players

    def test_basic_import(self) -> None:
        conn, repo, players = self._setup()
        rows = [
            {"player_name": "Mike Trout", "team_name": "Team A", "cost": "25"},
            {"player_name": "Shohei Ohtani", "team_name": "Team B", "cost": "30"},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 2
        assert result.skipped == 0
        assert result.unmatched == []

        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 2
        conn.close()

    def test_unmatched_name(self) -> None:
        conn, repo, players = self._setup()
        rows = [
            {"player_name": "Mike Trout", "team_name": "Team A"},
            {"player_name": "Nonexistent Player", "team_name": "Team B"},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        assert result.unmatched == ["Nonexistent Player"]
        conn.close()

    def test_missing_team_name_skipped(self) -> None:
        conn, repo, players = self._setup()
        rows = [
            {"player_name": "Mike Trout", "team_name": ""},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty")

        assert result.loaded == 0
        assert result.skipped == 1
        conn.close()

    def test_missing_player_name_skipped(self) -> None:
        conn, repo, players = self._setup()
        rows = [
            {"player_name": "", "team_name": "Team A"},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty")

        assert result.loaded == 0
        assert result.skipped == 1
        conn.close()

    def test_cost_with_dollar_sign(self) -> None:
        conn, repo, players = self._setup()
        rows = [
            {"player_name": "Mike Trout", "team_name": "Team A", "cost": "$25"},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        stored = repo.find_by_season_league(2026, "dynasty")
        assert stored[0].cost == 25.0
        conn.close()

    def test_no_cost(self) -> None:
        conn, repo, players = self._setup()
        rows = [
            {"player_name": "Mike Trout", "team_name": "Team A"},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        stored = repo.find_by_season_league(2026, "dynasty")
        assert stored[0].cost is None
        conn.close()

    def test_alternate_column_names(self) -> None:
        conn, repo, players = self._setup()
        rows = [
            {"Player": "Mike Trout", "Team": "Team A", "Cost": "25"},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty")
        conn.commit()

        assert result.loaded == 1
        stored = repo.find_by_season_league(2026, "dynasty")
        assert stored[0].team_name == "Team A"
        assert stored[0].cost == 25.0
        conn.close()
