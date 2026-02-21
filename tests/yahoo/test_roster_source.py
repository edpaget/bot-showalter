import datetime
import sqlite3
from typing import Any

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.yahoo_player_map_repo import SqliteYahooPlayerMapRepo
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper
from fantasy_baseball_manager.yahoo.roster_source import YahooRosterSource


class FakeClient:
    def __init__(self, roster_response: dict[str, Any] | None = None) -> None:
        self._roster_response = roster_response or _DEFAULT_ROSTER_RESPONSE

    def get_roster(self, team_key: str) -> dict[str, Any]:
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


class TestYahooRosterSource:
    def test_parse_roster_entries(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
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
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
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
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
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
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
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
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
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
