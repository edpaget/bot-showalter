import datetime

from fantasy_baseball_manager.domain.transaction import Transaction, TransactionPlayer


class TestTransaction:
    def test_construction(self) -> None:
        txn = Transaction(
            transaction_key="449.l.12345.tr.1",
            league_key="449.l.12345",
            type="add",
            timestamp=datetime.datetime(2026, 3, 1, 12, 0, 0, tzinfo=datetime.UTC),
            status="successful",
            trader_team_key="449.l.12345.t.1",
        )
        assert txn.transaction_key == "449.l.12345.tr.1"
        assert txn.type == "add"
        assert txn.status == "successful"

    def test_frozen(self) -> None:
        txn = Transaction(
            transaction_key="449.l.12345.tr.1",
            league_key="449.l.12345",
            type="add",
            timestamp=datetime.datetime(2026, 3, 1, tzinfo=datetime.UTC),
            status="successful",
            trader_team_key="449.l.12345.t.1",
        )
        try:
            txn.type = "drop"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_optional_fields_default_none(self) -> None:
        txn = Transaction(
            transaction_key="449.l.12345.tr.1",
            league_key="449.l.12345",
            type="add",
            timestamp=datetime.datetime(2026, 3, 1, tzinfo=datetime.UTC),
            status="successful",
            trader_team_key="449.l.12345.t.1",
        )
        assert txn.tradee_team_key is None
        assert txn.id is None

    def test_with_tradee_team_key(self) -> None:
        txn = Transaction(
            transaction_key="449.l.12345.tr.1",
            league_key="449.l.12345",
            type="trade",
            timestamp=datetime.datetime(2026, 3, 1, tzinfo=datetime.UTC),
            status="successful",
            trader_team_key="449.l.12345.t.1",
            tradee_team_key="449.l.12345.t.2",
        )
        assert txn.tradee_team_key == "449.l.12345.t.2"


class TestTransactionPlayer:
    def test_construction(self) -> None:
        tp = TransactionPlayer(
            transaction_key="449.l.12345.tr.1",
            player_id=42,
            yahoo_player_key="449.p.12345",
            player_name="Mike Trout",
            source_team_key=None,
            dest_team_key="449.l.12345.t.1",
            type="add",
        )
        assert tp.player_name == "Mike Trout"
        assert tp.type == "add"

    def test_frozen(self) -> None:
        tp = TransactionPlayer(
            transaction_key="449.l.12345.tr.1",
            player_id=42,
            yahoo_player_key="449.p.12345",
            player_name="Mike Trout",
            source_team_key=None,
            dest_team_key="449.l.12345.t.1",
            type="add",
        )
        try:
            tp.type = "drop"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_optional_fields_default_none(self) -> None:
        tp = TransactionPlayer(
            transaction_key="449.l.12345.tr.1",
            player_id=None,
            yahoo_player_key="449.p.12345",
            player_name="Unknown",
            source_team_key=None,
            dest_team_key=None,
            type="drop",
        )
        assert tp.player_id is None
        assert tp.source_team_key is None
        assert tp.dest_team_key is None
        assert tp.id is None
