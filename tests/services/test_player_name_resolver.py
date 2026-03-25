from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.identity import PlayerIdentity, PlayerType
from fantasy_baseball_manager.domain.player_alias import PlayerAlias
from fantasy_baseball_manager.repos.player_alias_repo import SqlitePlayerAliasRepo
from fantasy_baseball_manager.services.player_name_resolver import PlayerNameResolver
from tests.helpers import seed_player


def _setup() -> tuple[PlayerNameResolver, SqlitePlayerAliasRepo]:
    conn = create_connection(":memory:")
    seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
    seed_player(conn, player_id=2, name_first="Bobby", name_last="Witt")
    seed_player(conn, player_id=3, name_first="Bobby", name_last="Witt")
    seed_player(conn, player_id=4, name_first="Shohei", name_last="Ohtani")
    provider = SingleConnectionProvider(conn)
    repo = SqlitePlayerAliasRepo(provider)
    resolver = PlayerNameResolver(repo)
    return resolver, repo


class TestResolveExactMatch:
    def test_single_match(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="mike trout", player_id=1, player_type=PlayerType.BATTER))

        result = resolver.resolve("Mike Trout", player_type=PlayerType.BATTER)

        assert result == PlayerIdentity(1, PlayerType.BATTER)

    def test_normalization_strips_accents(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="jose ramirez", player_id=1, player_type=PlayerType.BATTER))

        result = resolver.resolve("José Ramírez", player_type=PlayerType.BATTER)

        assert result is not None
        assert result.player_id == 1

    def test_normalization_strips_suffix(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="ronald acuna", player_id=1, player_type=PlayerType.BATTER))

        result = resolver.resolve("Ronald Acuña Jr.", player_type=PlayerType.BATTER)

        assert result is not None
        assert result.player_id == 1

    def test_normalization_applies_nick_alias(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="matt boyd", player_id=1, player_type=PlayerType.PITCHER))

        result = resolver.resolve("Matthew Boyd", player_type=PlayerType.PITCHER)

        assert result is not None
        assert result.player_id == 1


class TestResolveNoMatch:
    def test_returns_none(self) -> None:
        resolver, _ = _setup()

        assert resolver.resolve("Nonexistent Player") is None


class TestSeasonDisambiguation:
    def test_bobby_witt_father_vs_jr(self) -> None:
        resolver, repo = _setup()
        # Father: active 1986-2004
        repo.upsert(
            PlayerAlias(
                alias_name="bobby witt",
                player_id=2,
                player_type=PlayerType.BATTER,
                active_from=1986,
                active_to=2004,
            )
        )
        # Son: active 2021-present
        repo.upsert(
            PlayerAlias(
                alias_name="bobby witt",
                player_id=3,
                player_type=PlayerType.BATTER,
                active_from=2021,
                active_to=None,
            )
        )

        result_modern = resolver.resolve("Bobby Witt", season=2025, player_type=PlayerType.BATTER)
        assert result_modern is not None
        assert result_modern.player_id == 3

        result_historical = resolver.resolve("Bobby Witt", season=2000, player_type=PlayerType.BATTER)
        assert result_historical is not None
        assert result_historical.player_id == 2


class TestPlayerTypeFilter:
    def test_filters_by_type(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="shohei ohtani", player_id=4, player_type=PlayerType.BATTER))
        repo.upsert(PlayerAlias(alias_name="shohei ohtani", player_id=4, player_type=PlayerType.PITCHER))

        batter = resolver.resolve("Shohei Ohtani", player_type=PlayerType.BATTER)
        assert batter == PlayerIdentity(4, PlayerType.BATTER)

        pitcher = resolver.resolve("Shohei Ohtani", player_type=PlayerType.PITCHER)
        assert pitcher == PlayerIdentity(4, PlayerType.PITCHER)

    def test_untyped_alias_matches_with_caller_type(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="mike trout", player_id=1, player_type=None))

        result = resolver.resolve("Mike Trout", player_type=PlayerType.BATTER)

        assert result == PlayerIdentity(1, PlayerType.BATTER)


class TestAmbiguous:
    def test_ambiguous_returns_none(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="bobby witt", player_id=2, player_type=PlayerType.BATTER))
        repo.upsert(PlayerAlias(alias_name="bobby witt", player_id=3, player_type=PlayerType.BATTER))

        # No season hint to disambiguate
        result = resolver.resolve("Bobby Witt", player_type=PlayerType.BATTER)
        assert result is None

    def test_no_type_no_caller_type_returns_none(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="mike trout", player_id=1, player_type=None))

        result = resolver.resolve("Mike Trout")
        assert result is None


class TestRegisterAlias:
    def test_registers_and_resolves(self) -> None:
        resolver, _ = _setup()
        identity = PlayerIdentity(1, PlayerType.BATTER)

        resolver.register_alias("m trout", identity, source="manual")
        result = resolver.resolve("m trout", player_type=PlayerType.BATTER)

        assert result == identity


class TestResolveAll:
    def test_returns_all_candidates(self) -> None:
        resolver, repo = _setup()
        repo.upsert(PlayerAlias(alias_name="bobby witt", player_id=2, player_type=PlayerType.BATTER))
        repo.upsert(PlayerAlias(alias_name="bobby witt", player_id=3, player_type=PlayerType.BATTER))

        results = resolver.resolve_all("Bobby Witt")

        assert len(results) == 2
        ids = {r.player_id for r in results}
        assert ids == {2, 3}

    def test_season_filter_narrows(self) -> None:
        resolver, repo = _setup()
        repo.upsert(
            PlayerAlias(
                alias_name="bobby witt",
                player_id=2,
                player_type=PlayerType.BATTER,
                active_from=1986,
                active_to=2004,
            )
        )
        repo.upsert(
            PlayerAlias(
                alias_name="bobby witt",
                player_id=3,
                player_type=PlayerType.BATTER,
                active_from=2021,
                active_to=None,
            )
        )

        results = resolver.resolve_all("Bobby Witt", season=2025)
        assert len(results) == 1
        assert results[0].player_id == 3
