import datetime
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.yahoo_player_map_repo import SqliteYahooPlayerMapRepo
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper
from fantasy_baseball_manager.yahoo.transaction_source import YahooTransactionSource

if TYPE_CHECKING:
    import sqlite3


class FakeClient:
    def __init__(self, transactions_response: dict[str, Any]) -> None:
        self._transactions_response = transactions_response

    def get_transactions(self, league_key: str) -> dict[str, Any]:
        return self._transactions_response


def _player_meta(
    player_key: str,
    name: str,
    mlbam_id: int,
    txn_type: str,
    *,
    source_team: str | None = None,
    dest_team: str | None = None,
) -> dict[str, Any]:
    """Build a single player entry in Yahoo transaction format."""
    txn_data: dict[str, str] = {"type": txn_type}
    if source_team:
        txn_data["source_team_key"] = source_team
    if dest_team:
        txn_data["destination_team_key"] = dest_team
    return {
        "player": [
            [
                {"player_key": player_key},
                {"name": {"full": name}},
                {"editorial_team_abbr": "LAA"},
                {"eligible_positions": [{"position": "CF"}]},
                {"player_id": mlbam_id},
            ],
            {"transaction_data": txn_data},
        ]
    }


def _build_response(*transactions: list[Any]) -> dict[str, Any]:
    """Wrap transaction dicts in Yahoo response structure."""
    txns: dict[str, Any] = {"count": len(transactions)}
    for i, txn in enumerate(transactions):
        txns[str(i)] = {"transaction": txn}
    return {
        "fantasy_content": {
            "league": [
                {"league_key": "449.l.12345"},
                {"transactions": txns},
            ]
        }
    }


def _add_drop_transaction() -> list[Any]:
    """An add/drop transaction: add Mike Trout, drop John Doe."""
    return [
        {
            "transaction_key": "449.l.12345.tr.1",
            "type": "add/drop",
            "timestamp": "1709290800",  # 2024-03-01 12:00 UTC (approx)
            "status": "successful",
            "trader_team_key": "449.l.12345.t.1",
        },
        {
            "players": {
                "0": _player_meta(
                    "449.p.11111",
                    "Mike Trout",
                    545361,
                    "add",
                    dest_team="449.l.12345.t.1",
                ),
                "1": _player_meta(
                    "449.p.22222",
                    "John Doe",
                    999999,
                    "drop",
                    source_team="449.l.12345.t.1",
                ),
                "count": 2,
            }
        },
    ]


def _trade_transaction() -> list[Any]:
    """A trade transaction between two teams."""
    return [
        {
            "transaction_key": "449.l.12345.tr.2",
            "type": "trade",
            "timestamp": "1709377200",  # ~2024-03-02 12:00 UTC
            "status": "successful",
            "trader_team_key": "449.l.12345.t.1",
            "tradee_team_key": "449.l.12345.t.2",
        },
        {
            "players": {
                "0": _player_meta(
                    "449.p.11111",
                    "Mike Trout",
                    545361,
                    "add",
                    source_team="449.l.12345.t.1",
                    dest_team="449.l.12345.t.2",
                ),
                "1": _player_meta(
                    "449.p.33333",
                    "Aaron Judge",
                    592450,
                    "add",
                    source_team="449.l.12345.t.2",
                    dest_team="449.l.12345.t.1",
                ),
                "count": 2,
            }
        },
    ]


def _waiver_transaction() -> list[Any]:
    """A waiver claim transaction."""
    return [
        {
            "transaction_key": "449.l.12345.tr.3",
            "type": "waiver",
            "timestamp": "1709463600",  # ~2024-03-03 12:00 UTC
            "status": "successful",
            "trader_team_key": "449.l.12345.t.3",
        },
        {
            "players": {
                "0": _player_meta(
                    "449.p.44444",
                    "Shohei Ohtani",
                    660271,
                    "add",
                    dest_team="449.l.12345.t.3",
                ),
                "count": 1,
            }
        },
    ]


class TestYahooTransactionSource:
    def test_parse_add_drop(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        player_repo.upsert(Player(name_first="John", name_last="Doe", mlbam_id=999999))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        response = _build_response(_add_drop_transaction())
        client = FakeClient(response)
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]

        result = source.fetch_transactions("449.l.12345")
        assert len(result) == 1

        txn, players = result[0]
        assert txn.type == "add"  # add/drop normalizes to "add"
        assert txn.status == "successful"
        assert txn.trader_team_key == "449.l.12345.t.1"
        assert len(players) == 2

        add_player = next(p for p in players if p.type == "add")
        assert add_player.player_name == "Mike Trout"
        assert add_player.dest_team_key == "449.l.12345.t.1"

        drop_player = next(p for p in players if p.type == "drop")
        assert drop_player.player_name == "John Doe"
        assert drop_player.source_team_key == "449.l.12345.t.1"

    def test_parse_trade(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        player_repo.upsert(Player(name_first="Aaron", name_last="Judge", mlbam_id=592450))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        response = _build_response(_trade_transaction())
        client = FakeClient(response)
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]

        result = source.fetch_transactions("449.l.12345")
        assert len(result) == 1

        txn, players = result[0]
        assert txn.type == "trade"
        assert txn.trader_team_key == "449.l.12345.t.1"
        assert txn.tradee_team_key == "449.l.12345.t.2"
        assert len(players) == 2

    def test_parse_waiver(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        player_repo.upsert(Player(name_first="Shohei", name_last="Ohtani", mlbam_id=660271))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        response = _build_response(_waiver_transaction())
        client = FakeClient(response)
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]

        result = source.fetch_transactions("449.l.12345")
        assert len(result) == 1

        txn, players = result[0]
        assert txn.type == "waiver"
        assert len(players) == 1
        assert players[0].player_name == "Shohei Ohtani"

    def test_incremental_fetch_filters_by_since(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        response = _build_response(
            _add_drop_transaction(),
            _trade_transaction(),
            _waiver_transaction(),
        )
        client = FakeClient(response)
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]

        # Since is after the add/drop but before the trade
        since = datetime.datetime(2024, 3, 1, 13, 0, 0, tzinfo=datetime.UTC)
        result = source.fetch_transactions("449.l.12345", since=since)
        # Should only include trade and waiver (timestamps after since)
        assert len(result) == 2
        types = {txn.type for txn, _ in result}
        assert "add" not in types  # the add/drop has timestamp before since

    def test_empty_response(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        mapper = YahooPlayerMapper(map_repo, player_repo)
        client = FakeClient({"fantasy_content": {"league": [{}]}})
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]

        result = source.fetch_transactions("449.l.12345")
        assert result == []

    def test_player_resolution(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        trout_id = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        mapper = YahooPlayerMapper(map_repo, player_repo)
        response = _build_response(_add_drop_transaction())
        client = FakeClient(response)
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]

        result = source.fetch_transactions("449.l.12345")
        _, players = result[0]

        # Trout should be resolved
        trout = next(p for p in players if p.player_name == "Mike Trout")
        assert trout.player_id == trout_id

        # John Doe has no matching player (mlbam 999999 not in DB)
        doe = next(p for p in players if p.player_name == "John Doe")
        assert doe.player_id is None

    def test_malformed_response_returns_empty(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        mapper = YahooPlayerMapper(map_repo, player_repo)

        client = FakeClient({})
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]
        assert source.fetch_transactions("449.l.12345") == []

        client = FakeClient({"fantasy_content": {"league": "bad"}})
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]
        assert source.fetch_transactions("449.l.12345") == []

    def test_pending_trade_normalizes_to_trade(self, conn: sqlite3.Connection) -> None:
        player_repo = SqlitePlayerRepo(conn)
        map_repo = SqliteYahooPlayerMapRepo(conn)
        mapper = YahooPlayerMapper(map_repo, player_repo)

        trade_txn = _trade_transaction()
        trade_txn[0]["type"] = "pending_trade"
        response = _build_response(trade_txn)
        client = FakeClient(response)
        source = YahooTransactionSource(client, mapper)  # type: ignore[arg-type]

        result = source.fetch_transactions("449.l.12345")
        assert result[0][0].type == "trade"
