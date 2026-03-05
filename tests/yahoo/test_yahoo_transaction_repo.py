import datetime
from typing import TYPE_CHECKING

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.transaction import Transaction, TransactionPlayer
from fantasy_baseball_manager.repos.yahoo_transaction_repo import (
    SqliteYahooTransactionRepo,
)

if TYPE_CHECKING:
    import sqlite3


def _make_transaction(**overrides: object) -> Transaction:
    defaults: dict[str, object] = {
        "transaction_key": "449.l.12345.tr.1",
        "league_key": "449.l.12345",
        "type": "add",
        "timestamp": datetime.datetime(2026, 3, 1, 12, 0, 0, tzinfo=datetime.UTC),
        "status": "successful",
        "trader_team_key": "449.l.12345.t.1",
    }
    defaults.update(overrides)
    return Transaction(**defaults)  # type: ignore[arg-type]


def _make_player(**overrides: object) -> TransactionPlayer:
    defaults: dict[str, object] = {
        "transaction_key": "449.l.12345.tr.1",
        "player_id": 42,
        "yahoo_player_key": "449.p.12345",
        "player_name": "Mike Trout",
        "source_team_key": None,
        "dest_team_key": "449.l.12345.t.1",
        "type": "add",
    }
    defaults.update(overrides)
    return TransactionPlayer(**defaults)  # type: ignore[arg-type]


class TestSqliteYahooTransactionRepo:
    def test_upsert_and_retrieve(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        txn = _make_transaction()
        players = [_make_player()]

        repo.upsert(txn, players)

        result = repo.get_by_league("449.l.12345")
        assert len(result) == 1
        assert result[0].transaction_key == "449.l.12345.tr.1"
        assert result[0].type == "add"
        assert result[0].status == "successful"
        assert result[0].trader_team_key == "449.l.12345.t.1"
        assert result[0].id is not None

    def test_upsert_idempotent(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        txn = _make_transaction()
        players = [_make_player()]

        repo.upsert(txn, players)
        repo.upsert(txn, players)

        result = repo.get_by_league("449.l.12345")
        assert len(result) == 1

    def test_upsert_updates_status(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        txn = _make_transaction(status="pending")
        repo.upsert(txn, [_make_player()])

        updated = _make_transaction(status="successful")
        repo.upsert(updated, [_make_player()])

        result = repo.get_by_league("449.l.12345")
        assert len(result) == 1
        assert result[0].status == "successful"

    def test_get_by_league_with_since(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        old = _make_transaction(
            transaction_key="449.l.12345.tr.1",
            timestamp=datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC),
        )
        new = _make_transaction(
            transaction_key="449.l.12345.tr.2",
            timestamp=datetime.datetime(2026, 3, 1, tzinfo=datetime.UTC),
        )
        repo.upsert(old, [])
        repo.upsert(new, [_make_player(transaction_key="449.l.12345.tr.2")])

        since = datetime.datetime(2026, 2, 15, tzinfo=datetime.UTC)
        result = repo.get_by_league("449.l.12345", since=since)
        assert len(result) == 1
        assert result[0].transaction_key == "449.l.12345.tr.2"

    def test_get_latest_timestamp_empty(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        assert repo.get_latest_timestamp("449.l.12345") is None

    def test_get_latest_timestamp(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        t1 = _make_transaction(
            transaction_key="449.l.12345.tr.1",
            timestamp=datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC),
        )
        t2 = _make_transaction(
            transaction_key="449.l.12345.tr.2",
            timestamp=datetime.datetime(2026, 3, 15, tzinfo=datetime.UTC),
        )
        repo.upsert(t1, [])
        repo.upsert(t2, [])

        latest = repo.get_latest_timestamp("449.l.12345")
        assert latest is not None
        assert latest == datetime.datetime(2026, 3, 15, tzinfo=datetime.UTC)

    def test_get_recent_filters_by_days(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        # Old transaction (more than 30 days ago)
        old = _make_transaction(
            transaction_key="449.l.12345.tr.1",
            timestamp=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
        )
        # Recent transaction
        recent = _make_transaction(
            transaction_key="449.l.12345.tr.2",
            timestamp=datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=1),
        )
        repo.upsert(old, [])
        repo.upsert(recent, [_make_player(transaction_key="449.l.12345.tr.2")])

        result = repo.get_recent("449.l.12345", days=7)
        assert len(result) == 1
        txn, players = result[0]
        assert txn.transaction_key == "449.l.12345.tr.2"
        assert len(players) == 1

    def test_get_players_for_transaction(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        txn = _make_transaction()
        players = [
            _make_player(yahoo_player_key="449.p.111", player_name="Player A", type="add"),
            _make_player(yahoo_player_key="449.p.222", player_name="Player B", type="drop"),
        ]
        repo.upsert(txn, players)

        result = repo.get_players_for_transaction("449.l.12345.tr.1")
        assert len(result) == 2
        names = {p.player_name for p in result}
        assert names == {"Player A", "Player B"}

    def test_get_players_for_nonexistent_transaction(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        result = repo.get_players_for_transaction("nonexistent")
        assert result == []

    def test_tradee_team_key(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        txn = _make_transaction(
            type="trade",
            tradee_team_key="449.l.12345.t.2",
        )
        repo.upsert(txn, [])

        result = repo.get_by_league("449.l.12345")
        assert result[0].tradee_team_key == "449.l.12345.t.2"

    def test_nullable_player_id(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooTransactionRepo(SingleConnectionProvider(conn))
        txn = _make_transaction()
        players = [_make_player(player_id=None)]
        repo.upsert(txn, players)

        result = repo.get_players_for_transaction("449.l.12345.tr.1")
        assert result[0].player_id is None
