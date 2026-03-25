from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.player_alias import PlayerAlias
from fantasy_baseball_manager.repos.player_alias_repo import SqlitePlayerAliasRepo
from tests.helpers import seed_player


def _make_alias(**overrides: object) -> PlayerAlias:
    defaults: dict[str, object] = {
        "alias_name": "mike trout",
        "player_id": 1,
        "player_type": PlayerType.BATTER,
        "source": "seed",
        "active_from": 2011,
        "active_to": None,
    }
    defaults.update(overrides)
    return PlayerAlias(**defaults)  # type: ignore[arg-type]


class TestUpsertAndFindByName:
    def test_round_trip(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        repo.upsert(_make_alias())
        results = repo.find_by_name("mike trout")

        assert len(results) == 1
        assert results[0].alias_name == "mike trout"
        assert results[0].player_id == 1
        assert results[0].player_type is PlayerType.BATTER
        assert results[0].source == "seed"
        assert results[0].active_from == 2011
        assert results[0].active_to is None
        assert results[0].id is not None

    def test_upsert_idempotent(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        repo.upsert(_make_alias())
        repo.upsert(_make_alias(source="manual"))
        results = repo.find_by_name("mike trout")

        assert len(results) == 1
        assert results[0].source == "manual"

    def test_no_match(self) -> None:
        conn = create_connection(":memory:")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        assert repo.find_by_name("nonexistent") == []


class TestFindByPlayer:
    def test_returns_all_aliases(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        repo.upsert(_make_alias(alias_name="mike trout"))
        repo.upsert(_make_alias(alias_name="m trout"))
        results = repo.find_by_player(1)

        assert len(results) == 2
        names = {r.alias_name for r in results}
        assert names == {"mike trout", "m trout"}


class TestMultiplePlayersSameName:
    def test_same_name_different_players(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Bobby", name_last="Witt")
        seed_player(conn, player_id=2, name_first="Bobby", name_last="Witt")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        repo.upsert(_make_alias(alias_name="bobby witt", player_id=1, active_from=1986, active_to=2004))
        repo.upsert(_make_alias(alias_name="bobby witt", player_id=2, active_from=2021, active_to=None))
        results = repo.find_by_name("bobby witt")

        assert len(results) == 2
        ids = {r.player_id for r in results}
        assert ids == {1, 2}


class TestPlayerTypeDiscrimination:
    def test_same_name_player_different_types(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Shohei", name_last="Ohtani")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        repo.upsert(_make_alias(alias_name="shohei ohtani", player_id=1, player_type=PlayerType.BATTER))
        repo.upsert(_make_alias(alias_name="shohei ohtani", player_id=1, player_type=PlayerType.PITCHER))
        results = repo.find_by_name("shohei ohtani")

        assert len(results) == 2
        types = {r.player_type for r in results}
        assert types == {PlayerType.BATTER, PlayerType.PITCHER}

    def test_none_player_type(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        repo.upsert(_make_alias(player_type=None))
        results = repo.find_by_name("mike trout")

        assert len(results) == 1
        assert results[0].player_type is None


class TestDeleteByPlayer:
    def test_deletes_all_aliases(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        repo.upsert(_make_alias(alias_name="mike trout"))
        repo.upsert(_make_alias(alias_name="m trout"))
        deleted = repo.delete_by_player(1)

        assert deleted == 2
        assert repo.find_by_player(1) == []


class TestUpsertBatch:
    def test_inserts_multiple(self) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
        seed_player(conn, player_id=2, name_first="Shohei", name_last="Ohtani")
        repo = SqlitePlayerAliasRepo(SingleConnectionProvider(conn))

        aliases = [
            _make_alias(alias_name="mike trout", player_id=1),
            _make_alias(alias_name="shohei ohtani", player_id=2),
        ]
        count = repo.upsert_batch(aliases)

        assert count == 2
        assert len(repo.find_by_name("mike trout")) == 1
        assert len(repo.find_by_name("shohei ohtani")) == 1
