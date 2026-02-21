import datetime
import sqlite3

from fantasy_baseball_manager.domain.roster import Roster, RosterEntry
from fantasy_baseball_manager.repos.yahoo_roster_repo import SqliteYahooRosterRepo


def _make_entry(**overrides: object) -> RosterEntry:
    defaults: dict[str, object] = {
        "player_id": 42,
        "yahoo_player_key": "449.p.12345",
        "player_name": "Mike Trout",
        "position": "CF",
        "roster_status": "active",
        "acquisition_type": "draft",
    }
    defaults.update(overrides)
    return RosterEntry(**defaults)  # type: ignore[arg-type]


def _make_roster(**overrides: object) -> Roster:
    defaults: dict[str, object] = {
        "team_key": "449.l.12345.t.1",
        "league_key": "449.l.12345",
        "season": 2026,
        "week": 1,
        "as_of": datetime.date(2026, 3, 27),
        "entries": (_make_entry(),),
    }
    defaults.update(overrides)
    return Roster(**defaults)  # type: ignore[arg-type]


class TestSqliteYahooRosterRepo:
    def test_save_and_load_snapshot(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooRosterRepo(conn)
        roster = _make_roster()
        repo.save_snapshot(roster)

        result = repo.get_latest_by_team("449.l.12345.t.1", "449.l.12345")
        assert result is not None
        assert result.team_key == "449.l.12345.t.1"
        assert result.league_key == "449.l.12345"
        assert result.season == 2026
        assert result.week == 1
        assert result.as_of == datetime.date(2026, 3, 27)
        assert len(result.entries) == 1
        assert result.entries[0].player_name == "Mike Trout"
        assert result.entries[0].player_id == 42
        assert result.id is not None

    def test_get_latest_by_team_returns_none_for_missing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooRosterRepo(conn)
        assert repo.get_latest_by_team("nonexistent", "449.l.12345") is None

    def test_get_latest_by_team_returns_most_recent(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooRosterRepo(conn)
        repo.save_snapshot(_make_roster(week=1, as_of=datetime.date(2026, 3, 27)))
        repo.save_snapshot(
            _make_roster(
                week=2,
                as_of=datetime.date(2026, 4, 3),
                entries=(_make_entry(player_name="Aaron Judge"),),
            )
        )

        result = repo.get_latest_by_team("449.l.12345.t.1", "449.l.12345")
        assert result is not None
        assert result.week == 2
        assert result.entries[0].player_name == "Aaron Judge"

    def test_get_by_league_latest(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooRosterRepo(conn)
        repo.save_snapshot(
            _make_roster(
                team_key="449.l.12345.t.1",
                entries=(_make_entry(player_name="Mike Trout"),),
            )
        )
        repo.save_snapshot(
            _make_roster(
                team_key="449.l.12345.t.2",
                entries=(_make_entry(player_name="Aaron Judge"),),
            )
        )

        results = repo.get_by_league_latest("449.l.12345")
        assert len(results) == 2
        names = {r.entries[0].player_name for r in results}
        assert names == {"Mike Trout", "Aaron Judge"}

    def test_resync_replaces_entries(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooRosterRepo(conn)
        repo.save_snapshot(_make_roster(entries=(_make_entry(player_name="Old Player"),)))
        repo.save_snapshot(_make_roster(entries=(_make_entry(player_name="New Player"),)))

        result = repo.get_latest_by_team("449.l.12345.t.1", "449.l.12345")
        assert result is not None
        assert len(result.entries) == 1
        assert result.entries[0].player_name == "New Player"

    def test_nullable_player_id(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooRosterRepo(conn)
        entry = _make_entry(player_id=None, player_name="Unresolved Player")
        repo.save_snapshot(_make_roster(entries=(entry,)))

        result = repo.get_latest_by_team("449.l.12345.t.1", "449.l.12345")
        assert result is not None
        assert result.entries[0].player_id is None
        assert result.entries[0].player_name == "Unresolved Player"

    def test_multiple_entries_per_snapshot(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooRosterRepo(conn)
        entries = (
            _make_entry(yahoo_player_key="449.p.1", player_name="Player A"),
            _make_entry(yahoo_player_key="449.p.2", player_name="Player B"),
            _make_entry(yahoo_player_key="449.p.3", player_name="Player C"),
        )
        repo.save_snapshot(_make_roster(entries=entries))

        result = repo.get_latest_by_team("449.l.12345.t.1", "449.l.12345")
        assert result is not None
        assert len(result.entries) == 3
