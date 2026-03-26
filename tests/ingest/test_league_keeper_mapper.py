from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.player_alias import PlayerAlias
from fantasy_baseball_manager.ingest.league_keeper_mapper import import_league_keepers
from fantasy_baseball_manager.repos.league_keeper_repo import SqliteLeagueKeeperRepo
from fantasy_baseball_manager.repos.player_alias_repo import SqlitePlayerAliasRepo
from fantasy_baseball_manager.services.player_name_resolver import PlayerNameResolver
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


class TestImportLeagueKeepersWithResolver:
    def _setup_with_resolver(self) -> tuple:
        conn = create_connection(":memory:")
        p1 = seed_player(conn, name_first="Mike", name_last="Trout")
        p2 = seed_player(conn, name_first="Shohei", name_last="Ohtani")
        provider = SingleConnectionProvider(conn)
        keeper_repo = SqliteLeagueKeeperRepo(provider)
        alias_repo = SqlitePlayerAliasRepo(provider)
        resolver = PlayerNameResolver(alias_repo)
        players = [
            Player(id=p1, name_first="Mike", name_last="Trout"),
            Player(id=p2, name_first="Shohei", name_last="Ohtani"),
        ]
        # Seed aliases
        alias_repo.upsert(PlayerAlias(alias_name="mike trout", player_id=p1, player_type=PlayerType.BATTER))
        alias_repo.upsert(PlayerAlias(alias_name="shohei ohtani", player_id=p2, player_type=PlayerType.PITCHER))
        return conn, keeper_repo, alias_repo, resolver, players

    def test_resolver_matches_with_player_type(self) -> None:
        conn, repo, _alias_repo, resolver, players = self._setup_with_resolver()
        rows = [
            {"player_name": "Shohei Ohtani", "team_name": "Team B", "cost": "30", "player_type": "pitcher"},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty", resolver=resolver)
        conn.commit()

        assert result.loaded == 1
        assert result.unmatched == []
        stored = repo.find_by_season_league(2026, "dynasty")
        assert stored[0].player_id == players[1].id
        assert stored[0].player_type == PlayerType.PITCHER
        conn.close()

    def test_resolver_registers_alias_when_type_known(self) -> None:
        conn, repo, alias_repo, resolver, players = self._setup_with_resolver()
        rows = [
            {"player_name": "Mike Trout", "team_name": "Team A", "cost": "25", "player_type": "batter"},
        ]
        import_league_keepers(rows, repo, players, season=2026, league="dynasty", resolver=resolver)

        aliases = alias_repo.find_by_name("mike trout")
        sources = {a.source for a in aliases}
        assert "league_keeper" in sources
        conn.close()

    def test_resolver_skips_alias_when_type_unknown(self) -> None:
        conn, repo, alias_repo, resolver, players = self._setup_with_resolver()
        rows = [
            {"player_name": "Mike Trout", "team_name": "Team A", "cost": "25"},
        ]
        import_league_keepers(rows, repo, players, season=2026, league="dynasty", resolver=resolver)

        aliases = alias_repo.find_by_name("mike trout")
        assert all(a.source != "league_keeper" for a in aliases)
        conn.close()

    def test_resolver_unmatched_when_no_alias(self) -> None:
        conn, repo, _alias_repo, resolver, players = self._setup_with_resolver()
        rows = [
            {"player_name": "Nobody McFakerson", "team_name": "Team X", "cost": "5"},
        ]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty", resolver=resolver)

        assert result.loaded == 0
        assert result.unmatched == ["Nobody McFakerson"]
        conn.close()

    def test_resolver_none_falls_back_to_old_behavior(self) -> None:
        conn = create_connection(":memory:")
        p1 = seed_player(conn, name_first="Mike", name_last="Trout")
        repo = SqliteLeagueKeeperRepo(SingleConnectionProvider(conn))
        players = [Player(id=p1, name_first="Mike", name_last="Trout")]

        rows = [{"player_name": "Mike Trout", "team_name": "Team A", "cost": "25"}]
        result = import_league_keepers(rows, repo, players, season=2026, league="dynasty", resolver=None)
        conn.commit()

        assert result.loaded == 1
        conn.close()
