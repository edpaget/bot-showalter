from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.repos.player_alias_repo import SqlitePlayerAliasRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.services.alias_seeder import seed_aliases
from tests.helpers import seed_player


def _seed_stats(conn, player_id: int, table: str, seasons: list[int]) -> None:
    """Insert minimal stats rows for active-range derivation."""
    for season in seasons:
        if table == "batting_stats":
            conn.execute(
                "INSERT INTO batting_stats (player_id, season, source) VALUES (?, ?, ?)",
                (player_id, season, "fangraphs"),
            )
        else:
            conn.execute(
                "INSERT INTO pitching_stats (player_id, season, source) VALUES (?, ?, ?)",
                (player_id, season, "fangraphs"),
            )
    conn.commit()


class TestSeedGeneratesNameVariants:
    def test_first_last_normalized(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        _seed_stats(conn, 1, "batting_stats", [2011, 2025])
        provider = SingleConnectionProvider(conn)
        alias_repo = SqlitePlayerAliasRepo(provider)
        player_repo = SqlitePlayerRepo(provider)

        seed_aliases(player_repo, alias_repo, provider)

        aliases = alias_repo.find_by_player(1)
        names = {a.alias_name for a in aliases}
        assert "mike trout" in names
        # "last, first" variant (comma preserved by normalize_name)
        assert "trout, mike" in names
        # first-initial variant
        assert "m trout" in names


class TestSeedAppliesNickAliases:
    def test_matthew_becomes_matt(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Matthew", name_last="Boyd")
        _seed_stats(conn, 1, "pitching_stats", [2018, 2024])
        provider = SingleConnectionProvider(conn)
        alias_repo = SqlitePlayerAliasRepo(provider)
        player_repo = SqlitePlayerRepo(provider)

        seed_aliases(player_repo, alias_repo, provider)

        aliases = alias_repo.find_by_player(1)
        names = {a.alias_name for a in aliases}
        # normalize_name("Matthew Boyd") = "matt boyd" (via NICK_ALIASES)
        # The formal name variant "matthew boyd" should also be generated
        assert "matt boyd" in names
        assert "matthew boyd" in names


class TestSeedPopulatesActiveYears:
    def test_active_from_and_to(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        _seed_stats(conn, 1, "batting_stats", [2011, 2015, 2025])
        provider = SingleConnectionProvider(conn)
        alias_repo = SqlitePlayerAliasRepo(provider)
        player_repo = SqlitePlayerRepo(provider)

        seed_aliases(player_repo, alias_repo, provider)

        aliases = alias_repo.find_by_player(1)
        assert len(aliases) >= 1
        alias = aliases[0]
        assert alias.active_from == 2011
        assert alias.active_to == 2025


class TestSeedIdempotent:
    def test_run_twice_no_duplicates(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        _seed_stats(conn, 1, "batting_stats", [2011, 2025])
        provider = SingleConnectionProvider(conn)
        alias_repo = SqlitePlayerAliasRepo(provider)
        player_repo = SqlitePlayerRepo(provider)

        seed_aliases(player_repo, alias_repo, provider)
        seed_aliases(player_repo, alias_repo, provider)

        aliases = alias_repo.find_by_player(1)
        # Should not have duplicates
        name_type_pairs = [(a.alias_name, a.player_type) for a in aliases]
        assert len(name_type_pairs) == len(set(name_type_pairs))


class TestSeedTwoWayPlayer:
    def test_gets_both_types(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Shohei", name_last="Ohtani")
        _seed_stats(conn, 1, "batting_stats", [2021, 2025])
        _seed_stats(conn, 1, "pitching_stats", [2021, 2023])
        provider = SingleConnectionProvider(conn)
        alias_repo = SqlitePlayerAliasRepo(provider)
        player_repo = SqlitePlayerRepo(provider)

        seed_aliases(player_repo, alias_repo, provider)

        aliases = alias_repo.find_by_name("shohei ohtani")
        types = {a.player_type for a in aliases}
        assert PlayerType.BATTER in types
        assert PlayerType.PITCHER in types
