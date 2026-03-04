import datetime
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import Transaction, TransactionPlayer, YahooLeague
from fantasy_baseball_manager.repos import (
    SqliteYahooLeagueRepo,
    SqliteYahooTeamRepo,
    SqliteYahooTransactionRepo,
)
from fantasy_baseball_manager.services.yahoo_sync import sync_league_metadata, sync_transactions
from fantasy_baseball_manager.yahoo.league_source import YahooLeagueSource

if TYPE_CHECKING:
    import sqlite3


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLeagueClient:
    """Fake client for league source."""

    def get_league_settings(self, league_key: str) -> dict[str, Any]:
        return {
            "fantasy_content": {
                "league": [
                    {
                        "league_key": "449.l.100",
                        "league_id": "100",
                        "name": "Test League",
                        "season": "2026",
                        "num_teams": 10,
                        "game_code": "mlb",
                    },
                    {
                        "settings": [
                            {
                                "draft_type": "live_standard_draft",
                            }
                        ]
                    },
                ]
            }
        }

    def get_teams(self, league_key: str) -> dict[str, Any]:
        return {
            "fantasy_content": {
                "league": [
                    {"league_key": "449.l.100"},
                    {
                        "teams": {
                            "0": {
                                "team": [
                                    [
                                        {"team_key": "449.l.100.t.1"},
                                        {"team_id": "1"},
                                        {"name": "Team A"},
                                        [],
                                        {"managers": [{"manager": {"nickname": "Alice", "is_current_login": "1"}}]},
                                    ]
                                ]
                            },
                            "count": 1,
                        }
                    },
                ]
            }
        }


class FakeTransactionSource:
    """Fake transaction source that returns canned transactions."""

    def __init__(self, transactions: list[tuple[Transaction, list[TransactionPlayer]]]) -> None:
        self._transactions = transactions
        self.last_since: datetime.datetime | None = None

    def fetch_transactions(
        self,
        league_key: str,
        *,
        since: datetime.datetime | None = None,
    ) -> list[tuple[Transaction, list[TransactionPlayer]]]:
        self.last_since = since
        if since is not None:
            return [(t, p) for t, p in self._transactions if t.timestamp > since]
        return self._transactions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyncLeagueMetadata:
    def test_upserts_league_and_teams(self, conn: sqlite3.Connection) -> None:
        league_repo = SqliteYahooLeagueRepo(conn)
        team_repo = SqliteYahooTeamRepo(conn)
        source = YahooLeagueSource(FakeLeagueClient())  # type: ignore[arg-type]
        result = sync_league_metadata(
            league_source=source,
            league_repo=league_repo,
            team_repo=team_repo,
            league_key="449.l.100",
            game_key="449",
        )

        assert result.league_key == "449.l.100"
        assert result.name == "Test League"

        persisted = league_repo.get_by_league_key("449.l.100")
        assert persisted is not None
        assert persisted.name == "Test League"

        teams = team_repo.get_by_league_key("449.l.100")
        assert len(teams) == 1
        assert teams[0].name == "Team A"

    def test_returns_league_object(self, conn: sqlite3.Connection) -> None:
        league_repo = SqliteYahooLeagueRepo(conn)
        team_repo = SqliteYahooTeamRepo(conn)
        source = YahooLeagueSource(FakeLeagueClient())  # type: ignore[arg-type]
        league = sync_league_metadata(
            league_source=source,
            league_repo=league_repo,
            team_repo=team_repo,
            league_key="449.l.100",
            game_key="449",
            is_keeper=True,
        )

        assert isinstance(league, YahooLeague)
        assert league.is_keeper is True


class TestSyncTransactions:
    def test_upserts_new_transactions(self, conn: sqlite3.Connection) -> None:
        txn_repo = SqliteYahooTransactionRepo(conn)
        txn = Transaction(
            transaction_key="txn_1",
            league_key="449.l.100",
            type="add",
            timestamp=datetime.datetime(2026, 3, 1, tzinfo=datetime.UTC),
            status="successful",
            trader_team_key="449.l.100.t.1",
        )
        player = TransactionPlayer(
            transaction_key="txn_1",
            player_id=None,
            yahoo_player_key="449.p.1000",
            player_name="Mike Trout",
            source_team_key=None,
            dest_team_key="449.l.100.t.1",
            type="add",
        )
        source = FakeTransactionSource([(txn, [player])])

        count = sync_transactions(
            transaction_source=source,
            transaction_repo=txn_repo,
            league_key="449.l.100",
        )

        assert count == 1
        stored = txn_repo.get_by_league("449.l.100")
        assert len(stored) == 1
        assert stored[0].transaction_key == "txn_1"

    def test_incremental_fetch_uses_latest_timestamp(self, conn: sqlite3.Connection) -> None:
        txn_repo = SqliteYahooTransactionRepo(conn)

        # Seed an existing transaction
        existing = Transaction(
            transaction_key="txn_old",
            league_key="449.l.100",
            type="add",
            timestamp=datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC),
            status="successful",
            trader_team_key="449.l.100.t.1",
        )
        txn_repo.upsert(existing, [])
        conn.commit()

        new_txn = Transaction(
            transaction_key="txn_new",
            league_key="449.l.100",
            type="drop",
            timestamp=datetime.datetime(2026, 3, 1, tzinfo=datetime.UTC),
            status="successful",
            trader_team_key="449.l.100.t.1",
        )
        source = FakeTransactionSource([(new_txn, [])])

        count = sync_transactions(
            transaction_source=source,
            transaction_repo=txn_repo,
            league_key="449.l.100",
        )

        # Source should have been called with the latest timestamp
        assert source.last_since == datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC)
        assert count == 1
