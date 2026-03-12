import datetime
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import Roster, RosterEntry
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.yahoo_player_map_repo import SqliteYahooPlayerMapRepo
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper
from fantasy_baseball_manager.yahoo.roster_source import YahooRosterSource

if TYPE_CHECKING:
    import sqlite3


class FakeRosterRepo:
    def __init__(self, roster: Roster | None = None) -> None:
        self._roster = roster

    def get_latest_by_team(self, team_key: str, league_key: str) -> Roster | None:
        return self._roster


class FakeClient:
    def __init__(self, roster_response: dict[str, Any] | None = None) -> None:
        self._roster_response = roster_response or _DEFAULT_ROSTER_RESPONSE
        self.last_week: int | None = None

    def get_roster(self, team_key: str, *, week: int | None = None) -> dict[str, Any]:
        self.last_week = week
        return self._roster_response


_DEFAULT_ROSTER_RESPONSE: dict[str, Any] = {
    "fantasy_content": {
        "team": [
            [
                {"team_key": "449.l.12345.t.1"},
                {"name": "My Team"},
            ],
            {
                "roster": {
                    "0": {
                        "players": {
                            "0": {
                                "player": [
                                    [
                                        {"player_key": "449.p.11111"},
                                        {"name": {"full": "Mike Trout"}},
                                        {"editorial_team_abbr": "LAA"},
                                        {
                                            "eligible_positions": [
                                                {"position": "CF"},
                                                {"position": "LF"},
                                                {"position": "Util"},
                                            ]
                                        },
                                        {"player_id": 545361},
                                    ],
                                    {
                                        "selected_position": [
                                            {"position": "CF"},
                                        ]
                                    },
                                    {
                                        "transaction_data": {
                                            "type": "draft",
                                        }
                                    },
                                ]
                            },
                            "1": {
                                "player": [
                                    [
                                        {"player_key": "449.p.22222"},
                                        {"name": {"full": "Gerrit Cole"}},
                                        {"editorial_team_abbr": "NYY"},
                                        {"eligible_positions": [{"position": "SP"}]},
                                        {"player_id": 543037},
                                    ],
                                    {
                                        "selected_position": [
                                            {"position": "SP"},
                                        ]
                                    },
                                    {
                                        "transaction_data": {
                                            "type": "add",
                                        }
                                    },
                                ]
                            },
                            "2": {
                                "player": [
                                    [
                                        {"player_key": "449.p.33333"},
                                        {"name": {"full": "IL Player"}},
                                        {"editorial_team_abbr": "BOS"},
                                        {"eligible_positions": [{"position": "SS"}]},
                                    ],
                                    {
                                        "selected_position": [
                                            {"position": "IL"},
                                        ]
                                    },
                                    {
                                        "transaction_data": {
                                            "type": "trade",
                                        }
                                    },
                                ]
                            },
                            "3": {
                                "player": [
                                    [
                                        {"player_key": "449.p.44444"},
                                        {"name": {"full": "Bench Guy"}},
                                        {"editorial_team_abbr": "CHC"},
                                        {"eligible_positions": [{"position": "2B"}, {"position": "Util"}]},
                                    ],
                                    {
                                        "selected_position": [
                                            {"position": "BN"},
                                        ]
                                    },
                                    {
                                        "transaction_data": {
                                            "type": "draft",
                                        }
                                    },
                                ]
                            },
                            "count": 4,
                        }
                    }
                }
            },
        ]
    }
}


_EMPTY_PLAYERS_ROSTER_RESPONSE: dict[str, Any] = {
    "fantasy_content": {
        "team": [
            [
                {"team_key": "449.l.12345.t.1"},
                {"name": "My Team"},
            ],
            {"roster": {"0": {"players": []}}},
        ]
    }
}


_EMPTY_ROSTER_RESPONSE: dict[str, Any] = {
    "fantasy_content": {
        "team": [
            [
                {"team_key": "449.l.12345.t.1"},
                {"name": "My Team"},
            ],
            {},
        ]
    }
}


class TestYahooRosterSource:
    def test_parse_roster_entries(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        # Seed one player so we can test resolution
        player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        source = YahooRosterSource(FakeClient(), mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        assert roster.team_key == "449.l.12345.t.1"
        assert roster.league_key == "449.l.12345"
        assert roster.season == 2026
        assert roster.week == 1
        assert len(roster.entries) == 4

    def test_roster_status_mapping(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        source = YahooRosterSource(FakeClient(), mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        statuses = {e.player_name: e.roster_status for e in roster.entries}
        assert statuses["Mike Trout"] == "active"
        assert statuses["Gerrit Cole"] == "active"
        assert statuses["IL Player"] == "il"
        assert statuses["Bench Guy"] == "bench"

    def test_acquisition_type_parsing(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        source = YahooRosterSource(FakeClient(), mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        acq_types = {e.player_name: e.acquisition_type for e in roster.entries}
        assert acq_types["Mike Trout"] == "draft"
        assert acq_types["Gerrit Cole"] == "add"
        assert acq_types["IL Player"] == "trade"

    def test_resolved_player_has_player_id(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        player_id = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        source = YahooRosterSource(FakeClient(), mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        trout = next(e for e in roster.entries if e.player_name == "Mike Trout")
        assert trout.player_id == player_id

    def test_unresolved_player_has_none_player_id(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        source = YahooRosterSource(FakeClient(), mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        # IL Player has no mlbam_id in the response and no seeded player
        il_player = next(e for e in roster.entries if e.player_name == "IL Player")
        assert il_player.player_id is None

    def test_week_none_passes_none_to_client_and_sets_week_zero(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        fake_client = FakeClient()
        source = YahooRosterSource(fake_client, mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2025,
            as_of=datetime.date(2025, 10, 1),
        )

        assert fake_client.last_week is None
        assert roster.week == 0

    def test_week_value_passes_through_to_client(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        fake_client = FakeClient()
        source = YahooRosterSource(fake_client, mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2025,
            week=5,
            as_of=datetime.date(2025, 10, 1),
        )

        assert fake_client.last_week == 5
        assert roster.week == 5

    def test_empty_players_list_returns_empty_entries(self, conn: sqlite3.Connection) -> None:
        """Yahoo returns players as [] (not {}) when the roster has no players."""
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        source = YahooRosterSource(FakeClient(_EMPTY_PLAYERS_ROSTER_RESPONSE), mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        assert roster.entries == ()

    def test_empty_roster_returns_empty_entries(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        source = YahooRosterSource(FakeClient(_EMPTY_ROSTER_RESPONSE), mapper)  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        assert roster.entries == ()
        assert roster.team_key == "449.l.12345.t.1"

    def test_cached_roster_returned_when_available(self, conn: sqlite3.Connection) -> None:
        cached_roster = Roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
            entries=(
                RosterEntry(
                    player_id=1,
                    yahoo_player_key="449.p.99999",
                    player_name="Cached Player",
                    position="CF",
                    roster_status="active",
                    acquisition_type="draft",
                ),
            ),
        )
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        fake_client = FakeClient()
        source = YahooRosterSource(fake_client, mapper, roster_repo=FakeRosterRepo(cached_roster))  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        assert roster is cached_roster
        # Client should not have been called
        assert fake_client.last_week is None

    def test_cache_miss_falls_through_to_client(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        fake_client = FakeClient()
        source = YahooRosterSource(fake_client, mapper, roster_repo=FakeRosterRepo(None))  # type: ignore[arg-type]

        roster = source.fetch_team_roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 27),
        )

        # Should have fetched from client
        assert fake_client.last_week == 1
        assert len(roster.entries) == 4
