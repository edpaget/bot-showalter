import sqlite3

import pytest

from fantasy_baseball_manager.domain.player import Player, Team
from fantasy_baseball_manager.repos.errors import PlayerConflictError
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo


class TestSqlitePlayerRepo:
    def test_upsert_and_get_by_id(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        player = Player(name_first="Mike", name_last="Trout", mlbam_id=545361)
        player_id = repo.upsert(player)
        result = repo.get_by_id(player_id)
        assert result is not None
        assert result.name_first == "Mike"
        assert result.name_last == "Trout"
        assert result.mlbam_id == 545361

    def test_get_by_id_not_found(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        assert repo.get_by_id(999) is None

    def test_get_by_mlbam_id(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        result = repo.get_by_mlbam_id(545361)
        assert result is not None
        assert result.name_last == "Trout"

    def test_get_by_bbref_id(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, bbref_id="troutmi01"))
        result = repo.get_by_bbref_id("troutmi01")
        assert result is not None
        assert result.name_last == "Trout"

    def test_search_by_name(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        repo.upsert(Player(name_first="Shohei", name_last="Ohtani", mlbam_id=660271))
        results = repo.search_by_name("Trout")
        assert len(results) == 1
        assert results[0].name_last == "Trout"

    def test_search_by_name_partial(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        results = repo.search_by_name("ike")
        assert len(results) == 1

    def test_all(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        repo.upsert(Player(name_first="Shohei", name_last="Ohtani", mlbam_id=660271))
        results = repo.all()
        assert len(results) == 2

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, bats="R"))
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, bats="L"))
        result = repo.get_by_mlbam_id(545361)
        assert result is not None
        assert result.bats == "L"
        assert len(repo.all()) == 1

    def test_upsert_raises_conflict_on_fangraphs_id(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, fangraphs_id=10155))
        with pytest.raises(PlayerConflictError) as exc_info:
            repo.upsert(Player(name_first="Joe", name_last="Smith", mlbam_id=999999, fangraphs_id=10155))
        assert exc_info.value.conflicting_column == "fangraphs_id"
        assert exc_info.value.existing_player.mlbam_id == 545361
        assert exc_info.value.new_player.mlbam_id == 999999

    def test_upsert_raises_conflict_on_bbref_id(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, bbref_id="troutmi01"))
        with pytest.raises(PlayerConflictError) as exc_info:
            repo.upsert(Player(name_first="Joe", name_last="Smith", mlbam_id=999999, bbref_id="troutmi01"))
        assert exc_info.value.conflicting_column == "bbref_id"

    def test_upsert_raises_conflict_on_retro_id(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, retro_id="troum001"))
        with pytest.raises(PlayerConflictError) as exc_info:
            repo.upsert(Player(name_first="Joe", name_last="Smith", mlbam_id=999999, retro_id="troum001"))
        assert exc_info.value.conflicting_column == "retro_id"

    def test_upsert_null_secondary_ids_no_conflict(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        repo.upsert(Player(name_first="Joe", name_last="Smith", mlbam_id=999999))
        assert len(repo.all()) == 2

    def test_get_by_last_name_exact_match(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        results = repo.get_by_last_name("Trout")
        assert len(results) == 1
        assert results[0].name_last == "Trout"

    def test_get_by_last_name_case_insensitive(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        results = repo.get_by_last_name("trout")
        assert len(results) == 1
        assert results[0].name_last == "Trout"

    def test_get_by_last_name_no_partial(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        results = repo.get_by_last_name("Tro")
        assert len(results) == 0

    def test_get_by_last_name_multiple(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Joe", name_last="Smith", mlbam_id=100001))
        repo.upsert(Player(name_first="John", name_last="Smith", mlbam_id=100002))
        results = repo.get_by_last_name("Smith")
        assert len(results) == 2

    def test_get_by_last_name_no_match(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        results = repo.get_by_last_name("Nobody")
        assert len(results) == 0

    def test_get_by_ids(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        id1 = repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        repo.upsert(Player(name_first="Shohei", name_last="Ohtani", mlbam_id=660271))
        id3 = repo.upsert(Player(name_first="Aaron", name_last="Judge", mlbam_id=592450))
        results = repo.get_by_ids([id1, id3])
        assert len(results) == 2
        result_ids = {r.id for r in results}
        assert result_ids == {id1, id3}

    def test_get_by_ids_empty_list(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        results = repo.get_by_ids([])
        assert results == []

    def test_get_by_ids_missing_ids_ignored(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        id1 = repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        results = repo.get_by_ids([id1, 9999])
        assert len(results) == 1
        assert results[0].id == id1

    def test_upsert_update_raises_conflict_on_secondary_key_change(self, conn: sqlite3.Connection) -> None:
        repo = SqlitePlayerRepo(conn)
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, fangraphs_id=10155))
        repo.upsert(Player(name_first="Shohei", name_last="Ohtani", mlbam_id=660271, fangraphs_id=19755))
        with pytest.raises(PlayerConflictError) as exc_info:
            repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, fangraphs_id=19755))
        assert exc_info.value.conflicting_column == "fangraphs_id"


def test_player_conflict_error_message() -> None:
    new = Player(name_first="Joe", name_last="Smith", mlbam_id=999999)
    existing = Player(id=1, name_first="Mike", name_last="Trout", mlbam_id=545361)
    error = PlayerConflictError(new, existing, "fangraphs_id")
    msg = str(error)
    assert "fangraphs_id" in msg
    assert "999999" in msg
    assert "545361" in msg


class TestSqliteTeamRepo:
    def test_upsert_and_get_by_abbreviation(self, conn: sqlite3.Connection) -> None:
        repo = SqliteTeamRepo(conn)
        team = Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="W")
        repo.upsert(team)
        result = repo.get_by_abbreviation("LAD")
        assert result is not None
        assert result.name == "Los Angeles Dodgers"

    def test_get_by_abbreviation_not_found(self, conn: sqlite3.Connection) -> None:
        repo = SqliteTeamRepo(conn)
        assert repo.get_by_abbreviation("XXX") is None

    def test_all(self, conn: sqlite3.Connection) -> None:
        repo = SqliteTeamRepo(conn)
        repo.upsert(Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="W"))
        repo.upsert(Team(abbreviation="NYY", name="New York Yankees", league="AL", division="E"))
        assert len(repo.all()) == 2

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteTeamRepo(conn)
        repo.upsert(Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="W"))
        repo.upsert(Team(abbreviation="LAD", name="LA Dodgers", league="NL", division="W"))
        result = repo.get_by_abbreviation("LAD")
        assert result is not None
        assert result.name == "LA Dodgers"
        assert len(repo.all()) == 1
