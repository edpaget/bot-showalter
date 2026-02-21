import sqlite3

from fantasy_baseball_manager.domain.yahoo_league import YahooLeague, YahooTeam
from fantasy_baseball_manager.repos.yahoo_league_repo import SqliteYahooLeagueRepo, SqliteYahooTeamRepo


def _make_yahoo_league(**overrides: object) -> YahooLeague:
    defaults: dict[str, object] = {
        "league_key": "449.l.12345",
        "name": "Test League",
        "season": 2026,
        "num_teams": 12,
        "draft_type": "live_standard_draft",
        "is_keeper": False,
        "game_key": "449",
    }
    defaults.update(overrides)
    return YahooLeague(**defaults)  # type: ignore[arg-type]


def _make_yahoo_team(**overrides: object) -> YahooTeam:
    defaults: dict[str, object] = {
        "team_key": "449.l.12345.t.1",
        "league_key": "449.l.12345",
        "team_id": 1,
        "name": "My Team",
        "manager_name": "Manager",
        "is_owned_by_user": False,
    }
    defaults.update(overrides)
    return YahooTeam(**defaults)  # type: ignore[arg-type]


class TestSqliteYahooLeagueRepo:
    def test_upsert_and_get_by_league_key(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooLeagueRepo(conn)
        league = _make_yahoo_league()
        repo.upsert(league)
        result = repo.get_by_league_key("449.l.12345")
        assert result is not None
        assert result.league_key == "449.l.12345"
        assert result.name == "Test League"
        assert result.season == 2026
        assert result.num_teams == 12
        assert result.draft_type == "live_standard_draft"
        assert result.is_keeper is False
        assert result.game_key == "449"
        assert result.id is not None

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooLeagueRepo(conn)
        repo.upsert(_make_yahoo_league(name="Old Name"))
        repo.upsert(_make_yahoo_league(name="New Name"))
        result = repo.get_by_league_key("449.l.12345")
        assert result is not None
        assert result.name == "New Name"

    def test_get_by_league_key_returns_none_for_missing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooLeagueRepo(conn)
        assert repo.get_by_league_key("nonexistent") is None

    def test_get_all(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooLeagueRepo(conn)
        repo.upsert(_make_yahoo_league(league_key="449.l.100", name="League A"))
        repo.upsert(_make_yahoo_league(league_key="449.l.200", name="League B"))
        results = repo.get_all()
        assert len(results) == 2
        keys = {r.league_key for r in results}
        assert keys == {"449.l.100", "449.l.200"}

    def test_upsert_returns_row_id(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooLeagueRepo(conn)
        row_id = repo.upsert(_make_yahoo_league())
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_is_keeper_round_trips(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooLeagueRepo(conn)
        repo.upsert(_make_yahoo_league(is_keeper=True))
        result = repo.get_by_league_key("449.l.12345")
        assert result is not None
        assert result.is_keeper is True


class TestSqliteYahooTeamRepo:
    def _seed_league(self, conn: sqlite3.Connection) -> None:
        league_repo = SqliteYahooLeagueRepo(conn)
        league_repo.upsert(_make_yahoo_league())

    def test_upsert_and_get_by_league_key(self, conn: sqlite3.Connection) -> None:
        self._seed_league(conn)
        repo = SqliteYahooTeamRepo(conn)
        repo.upsert(_make_yahoo_team())
        results = repo.get_by_league_key("449.l.12345")
        assert len(results) == 1
        assert results[0].team_key == "449.l.12345.t.1"
        assert results[0].name == "My Team"
        assert results[0].manager_name == "Manager"
        assert results[0].is_owned_by_user is False
        assert results[0].id is not None

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        self._seed_league(conn)
        repo = SqliteYahooTeamRepo(conn)
        repo.upsert(_make_yahoo_team(name="Old Name"))
        repo.upsert(_make_yahoo_team(name="New Name"))
        results = repo.get_by_league_key("449.l.12345")
        assert len(results) == 1
        assert results[0].name == "New Name"

    def test_get_by_league_key_returns_empty_for_missing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTeamRepo(conn)
        assert repo.get_by_league_key("nonexistent") == []

    def test_get_user_team(self, conn: sqlite3.Connection) -> None:
        self._seed_league(conn)
        repo = SqliteYahooTeamRepo(conn)
        repo.upsert(_make_yahoo_team(team_key="449.l.12345.t.1", is_owned_by_user=False))
        repo.upsert(_make_yahoo_team(team_key="449.l.12345.t.2", team_id=2, is_owned_by_user=True, name="User Team"))
        result = repo.get_user_team("449.l.12345")
        assert result is not None
        assert result.name == "User Team"
        assert result.is_owned_by_user is True

    def test_get_user_team_returns_none_when_no_user_team(self, conn: sqlite3.Connection) -> None:
        self._seed_league(conn)
        repo = SqliteYahooTeamRepo(conn)
        repo.upsert(_make_yahoo_team(is_owned_by_user=False))
        assert repo.get_user_team("449.l.12345") is None

    def test_multiple_teams_per_league(self, conn: sqlite3.Connection) -> None:
        self._seed_league(conn)
        repo = SqliteYahooTeamRepo(conn)
        for i in range(1, 4):
            repo.upsert(
                _make_yahoo_team(
                    team_key=f"449.l.12345.t.{i}",
                    team_id=i,
                    name=f"Team {i}",
                )
            )
        results = repo.get_by_league_key("449.l.12345")
        assert len(results) == 3
