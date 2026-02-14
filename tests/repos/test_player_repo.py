import sqlite3

from fantasy_baseball_manager.domain.player import Player, Team
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
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, position="CF"))
        repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, position="DH"))
        result = repo.get_by_mlbam_id(545361)
        assert result is not None
        assert result.position == "DH"
        assert len(repo.all()) == 1


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
