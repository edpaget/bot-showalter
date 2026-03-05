from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.yahoo_player_map_repo import SqliteYahooPlayerMapRepo
from fantasy_baseball_manager.yahoo.draft_source import YahooDraftSource
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper

if TYPE_CHECKING:
    import sqlite3


class FakeClient:
    def __init__(
        self,
        draft_response: dict[str, Any],
        players_response: dict[str, Any] | None = None,
    ) -> None:
        self._draft_response = draft_response
        self._players_response = players_response or {"fantasy_content": {"league": [{}]}}

    def get_draft_results(self, league_key: str) -> dict[str, Any]:
        return self._draft_response

    def get_players(self, league_key: str, player_keys: list[str]) -> dict[str, Any]:
        return self._players_response


_SNAKE_DRAFT_RESPONSE: dict[str, Any] = {
    "fantasy_content": {
        "league": [
            {"league_key": "449.l.12345"},
            {
                "draft_results": {
                    "0": {
                        "draft_result": {
                            "pick": 1,
                            "round": 1,
                            "team_key": "449.l.12345.t.1",
                            "player_key": "449.p.11111",
                        }
                    },
                    "1": {
                        "draft_result": {
                            "pick": 2,
                            "round": 1,
                            "team_key": "449.l.12345.t.2",
                            "player_key": "449.p.22222",
                        }
                    },
                    "count": 2,
                }
            },
        ]
    }
}

_AUCTION_DRAFT_RESPONSE: dict[str, Any] = {
    "fantasy_content": {
        "league": [
            {"league_key": "449.l.12345"},
            {
                "draft_results": {
                    "0": {
                        "draft_result": {
                            "pick": 1,
                            "round": 1,
                            "team_key": "449.l.12345.t.1",
                            "player_key": "449.p.11111",
                            "cost": "55",
                        }
                    },
                    "count": 1,
                }
            },
        ]
    }
}

_EMPTY_DRAFT_RESPONSE: dict[str, Any] = {
    "fantasy_content": {
        "league": [
            {"league_key": "449.l.12345"},
            {
                "draft_results": {
                    "count": 0,
                }
            },
        ]
    }
}

_PLAYERS_RESPONSE: dict[str, Any] = {
    "fantasy_content": {
        "league": [
            {"league_key": "449.l.12345"},
            {
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
                                        {"position": "OF"},
                                        {"position": "Util"},
                                    ]
                                },
                                {"player_id": 545361},
                            ]
                        ]
                    },
                    "1": {
                        "player": [
                            [
                                {"player_key": "449.p.22222"},
                                {"name": {"full": "Gerrit Cole"}},
                                {"editorial_team_abbr": "NYY"},
                                {
                                    "eligible_positions": [
                                        {"position": "SP"},
                                    ]
                                },
                                {"player_id": 543037},
                            ]
                        ]
                    },
                    "count": 2,
                }
            },
        ]
    }
}


class TestYahooDraftSource:
    def test_snake_draft_results(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        player_repo.upsert(Player(name_first="Gerrit", name_last="Cole", mlbam_id=543037))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        client = FakeClient(_SNAKE_DRAFT_RESPONSE, _PLAYERS_RESPONSE)
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]

        picks = source.fetch_draft_results("449.l.12345", 2026)
        assert len(picks) == 2
        assert picks[0].round == 1
        assert picks[0].pick == 1
        assert picks[0].player_name == "Mike Trout"
        assert picks[0].position == "CF"
        assert picks[0].cost is None
        assert picks[1].player_name == "Gerrit Cole"
        assert picks[1].position == "SP"

    def test_auction_draft_results_with_cost(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        client = FakeClient(_AUCTION_DRAFT_RESPONSE, _PLAYERS_RESPONSE)
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]

        picks = source.fetch_draft_results("449.l.12345", 2026)
        assert len(picks) == 1
        assert picks[0].cost == 55
        assert picks[0].player_name == "Mike Trout"

    def test_player_mapping(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        trout_id = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        client = FakeClient(_SNAKE_DRAFT_RESPONSE, _PLAYERS_RESPONSE)
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]

        picks = source.fetch_draft_results("449.l.12345", 2026)
        assert picks[0].player_id == trout_id

    def test_empty_results(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)
        client = FakeClient(_EMPTY_DRAFT_RESPONSE)
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]

        picks = source.fetch_draft_results("449.l.12345", 2026)
        assert picks == []

    def test_malformed_response_returns_empty(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        mapper = YahooPlayerMapper(map_repo, player_repo)

        # Missing fantasy_content entirely
        client = FakeClient({})
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]
        assert source.fetch_draft_results("449.l.12345", 2026) == []

        # league is not a list
        client = FakeClient({"fantasy_content": {"league": "bad"}})
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]
        assert source.fetch_draft_results("449.l.12345", 2026) == []

        # league list too short
        client = FakeClient({"fantasy_content": {"league": [{}]}})
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]
        assert source.fetch_draft_results("449.l.12345", 2026) == []

        # No draft_results key
        client = FakeClient({"fantasy_content": {"league": [{}, {"other": "data"}]}})
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]
        assert source.fetch_draft_results("449.l.12345", 2026) == []

    def test_unmapped_player_has_none_id(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        map_repo = SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn))
        # Don't seed any players — they'll be unresolvable
        mapper = YahooPlayerMapper(map_repo, player_repo)
        client = FakeClient(_SNAKE_DRAFT_RESPONSE, _PLAYERS_RESPONSE)
        source = YahooDraftSource(client, mapper)  # type: ignore[arg-type]

        picks = source.fetch_draft_results("449.l.12345", 2026)
        assert len(picks) == 2
        # Still has name and position from Yahoo data
        assert picks[0].player_name == "Mike Trout"
        assert picks[0].player_id is None
