import datetime
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import Transaction, TransactionPlayer

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteYahooTransactionRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, txn: Transaction, players: list[TransactionPlayer]) -> None:
        with self._provider.connection() as conn:
            conn.execute(
                "INSERT INTO yahoo_transaction"
                "    (transaction_key, league_key, type, timestamp, status,"
                "     trader_team_key, tradee_team_key)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(transaction_key) DO UPDATE SET"
                "    status=excluded.status",
                (
                    txn.transaction_key,
                    txn.league_key,
                    txn.type,
                    txn.timestamp.isoformat(),
                    txn.status,
                    txn.trader_team_key,
                    txn.tradee_team_key,
                ),
            )

            # Delete existing players and re-insert
            conn.execute(
                "DELETE FROM yahoo_transaction_player WHERE transaction_key = ?",
                (txn.transaction_key,),
            )

            for player in players:
                conn.execute(
                    "INSERT INTO yahoo_transaction_player"
                    "    (transaction_key, player_id, yahoo_player_key, player_name,"
                    "     source_team_key, dest_team_key, type)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        player.transaction_key,
                        player.player_id,
                        player.yahoo_player_key,
                        player.player_name,
                        player.source_team_key,
                        player.dest_team_key,
                        player.type,
                    ),
                )

    def get_by_league(self, league_key: str, *, since: datetime.datetime | None = None) -> list[Transaction]:
        with self._provider.connection() as conn:
            if since is not None:
                rows = conn.execute(
                    "SELECT id, transaction_key, league_key, type, timestamp, status,"
                    "       trader_team_key, tradee_team_key"
                    " FROM yahoo_transaction"
                    " WHERE league_key = ? AND timestamp > ?"
                    " ORDER BY timestamp",
                    (league_key, since.isoformat()),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, transaction_key, league_key, type, timestamp, status,"
                    "       trader_team_key, tradee_team_key"
                    " FROM yahoo_transaction"
                    " WHERE league_key = ?"
                    " ORDER BY timestamp",
                    (league_key,),
                ).fetchall()
            return [self._row_to_transaction(row) for row in rows]

    def get_players_for_transaction(self, transaction_key: str) -> list[TransactionPlayer]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT id, transaction_key, player_id, yahoo_player_key, player_name,"
                "       source_team_key, dest_team_key, type"
                " FROM yahoo_transaction_player"
                " WHERE transaction_key = ?",
                (transaction_key,),
            ).fetchall()
            return [
                TransactionPlayer(
                    transaction_key=row["transaction_key"],
                    player_id=row["player_id"],
                    yahoo_player_key=row["yahoo_player_key"],
                    player_name=row["player_name"],
                    source_team_key=row["source_team_key"],
                    dest_team_key=row["dest_team_key"],
                    type=row["type"],
                    id=row["id"],
                )
                for row in rows
            ]

    def get_latest_timestamp(self, league_key: str) -> datetime.datetime | None:
        with self._provider.connection() as conn:
            row = conn.execute(
                "SELECT MAX(timestamp) AS max_ts FROM yahoo_transaction WHERE league_key = ?",
                (league_key,),
            ).fetchone()
            if row is None or row["max_ts"] is None:
                return None
            return datetime.datetime.fromisoformat(row["max_ts"])

    def get_recent(self, league_key: str, *, days: int) -> list[tuple[Transaction, list[TransactionPlayer]]]:
        cutoff = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(days=days)
        transactions = self.get_by_league(league_key, since=cutoff)
        result: list[tuple[Transaction, list[TransactionPlayer]]] = []
        for txn in transactions:
            players = self.get_players_for_transaction(txn.transaction_key)
            result.append((txn, players))
        return result

    @staticmethod
    def _row_to_transaction(row: sqlite3.Row) -> Transaction:
        return Transaction(
            transaction_key=row["transaction_key"],
            league_key=row["league_key"],
            type=row["type"],
            timestamp=datetime.datetime.fromisoformat(row["timestamp"]),
            status=row["status"],
            trader_team_key=row["trader_team_key"],
            tradee_team_key=row["tradee_team_key"],
            id=row["id"],
        )
