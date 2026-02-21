import datetime

import pytest

from fantasy_baseball_manager.domain.roster import Roster, RosterEntry


class TestRosterEntry:
    def test_construct(self) -> None:
        entry = RosterEntry(
            player_id=42,
            yahoo_player_key="449.p.12345",
            player_name="Mike Trout",
            position="CF",
            roster_status="active",
            acquisition_type="draft",
        )
        assert entry.player_id == 42
        assert entry.yahoo_player_key == "449.p.12345"
        assert entry.player_name == "Mike Trout"
        assert entry.position == "CF"
        assert entry.roster_status == "active"
        assert entry.acquisition_type == "draft"

    def test_player_id_nullable(self) -> None:
        entry = RosterEntry(
            player_id=None,
            yahoo_player_key="449.p.99999",
            player_name="Unknown",
            position="DH",
            roster_status="active",
            acquisition_type="add",
        )
        assert entry.player_id is None

    def test_frozen(self) -> None:
        entry = RosterEntry(
            player_id=42,
            yahoo_player_key="449.p.12345",
            player_name="Mike Trout",
            position="CF",
            roster_status="active",
            acquisition_type="draft",
        )
        with pytest.raises(AttributeError):
            entry.player_id = 99  # type: ignore[misc]


class TestRoster:
    def test_construct(self) -> None:
        entry = RosterEntry(
            player_id=42,
            yahoo_player_key="449.p.12345",
            player_name="Mike Trout",
            position="CF",
            roster_status="active",
            acquisition_type="draft",
        )
        roster = Roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
            entries=(entry,),
        )
        assert roster.team_key == "449.l.12345.t.1"
        assert roster.league_key == "449.l.12345"
        assert roster.season == 2026
        assert roster.week == 1
        assert roster.as_of == datetime.date(2026, 3, 27)
        assert len(roster.entries) == 1
        assert roster.entries[0].player_name == "Mike Trout"

    def test_optional_id(self) -> None:
        roster = Roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
            entries=(),
        )
        assert roster.id is None

    def test_frozen(self) -> None:
        roster = Roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
            entries=(),
        )
        with pytest.raises(AttributeError):
            roster.team_key = "other"  # type: ignore[misc]
