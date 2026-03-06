import queue
import threading
from typing import TYPE_CHECKING

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import ConnectionPool
from fantasy_baseball_manager.domain.yahoo_draft_pick import YahooDraftPick
from fantasy_baseball_manager.repos import SqlitePlayerRepo, SqliteYahooPlayerMapRepo
from fantasy_baseball_manager.yahoo.draft_poller import YahooDraftPoller
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper

if TYPE_CHECKING:
    from pathlib import Path


def _make_pick(round: int = 1, pick: int = 1, **overrides: object) -> YahooDraftPick:
    defaults: dict[str, object] = {
        "league_key": "449.l.12345",
        "season": 2026,
        "round": round,
        "pick": pick,
        "team_key": "449.l.12345.t.1",
        "yahoo_player_key": f"449.p.{pick}",
        "player_id": pick,
        "player_name": f"Player {pick}",
        "position": "OF",
    }
    defaults.update(overrides)
    return YahooDraftPick(**defaults)  # type: ignore[arg-type]


class FakeSource:
    def __init__(self) -> None:
        self.picks: list[YahooDraftPick] = []

    def fetch_draft_results(self, league_key: str, season: int) -> list[YahooDraftPick]:
        return list(self.picks)


class TestYahooDraftPoller:
    def test_detects_new_picks(self) -> None:
        source = FakeSource()
        pick_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        poller = YahooDraftPoller(
            source=source,
            league_key="449.l.12345",
            season=2026,
            interval=5.0,
            pick_queue=pick_queue,
        )

        # No picks yet
        new = poller.poll_once()
        assert new == []
        assert pick_queue.empty()

        # Two new picks arrive
        source.picks = [_make_pick(1, 1), _make_pick(1, 2)]
        new = poller.poll_once()
        assert len(new) == 2
        assert pick_queue.qsize() == 2

    def test_no_callback_on_no_change(self) -> None:
        source = FakeSource()
        source.picks = [_make_pick(1, 1)]
        pick_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        poller = YahooDraftPoller(
            source=source,
            league_key="449.l.12345",
            season=2026,
            interval=5.0,
            pick_queue=pick_queue,
        )

        # First poll finds the pick
        poller.poll_once()
        assert pick_queue.qsize() == 1

        # Second poll — same picks, nothing new
        new = poller.poll_once()
        assert new == []
        assert pick_queue.qsize() == 1  # no new items added

    def test_incremental_detection(self) -> None:
        source = FakeSource()
        source.picks = [_make_pick(1, 1)]
        pick_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        poller = YahooDraftPoller(
            source=source,
            league_key="449.l.12345",
            season=2026,
            interval=5.0,
            pick_queue=pick_queue,
        )

        poller.poll_once()
        assert pick_queue.qsize() == 1

        # One more pick arrives
        source.picks = [_make_pick(1, 1), _make_pick(1, 2)]
        new = poller.poll_once()
        assert len(new) == 1
        assert new[0].pick == 2
        assert pick_queue.qsize() == 2

    def test_ordered_emission(self) -> None:
        source = FakeSource()
        source.picks = [
            _make_pick(1, 1, player_name="First"),
            _make_pick(1, 2, player_name="Second"),
            _make_pick(1, 3, player_name="Third"),
        ]
        pick_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        poller = YahooDraftPoller(
            source=source,
            league_key="449.l.12345",
            season=2026,
            interval=5.0,
            pick_queue=pick_queue,
        )

        new = poller.poll_once()
        assert len(new) == 3

        items = []
        while not pick_queue.empty():
            items.append(pick_queue.get_nowait())
        assert [p.player_name for p in items] == ["First", "Second", "Third"]


class TestDraftPollerThreadSafety:
    def test_poll_once_from_background_thread_with_pool(self, tmp_path: Path) -> None:
        """ConnectionPool with check_same_thread=False allows cross-thread DB access."""
        db_path = tmp_path / "test.db"
        # Create schema via a normal connection, then close it
        setup_conn = create_connection(db_path)
        setup_conn.close()

        pool = ConnectionPool(db_path, size=1)
        try:
            mapper = YahooPlayerMapper(
                SqliteYahooPlayerMapRepo(pool),
                SqlitePlayerRepo(pool),
            )

            class PoolBackedSource:
                """Source that exercises mapper.resolve() to trigger DB queries."""

                def __init__(self, _mapper: YahooPlayerMapper) -> None:
                    self._mapper = _mapper
                    self.picks: list[YahooDraftPick] = []

                def fetch_draft_results(self, league_key: str, season: int) -> list[YahooDraftPick]:
                    # Trigger a DB query from whatever thread calls this
                    self._mapper.resolve({"player_key": "449.p.9999", "name": "Nobody Real"})
                    return list(self.picks)

            source = PoolBackedSource(mapper)
            source.picks = [_make_pick(1, 1)]
            pick_queue: queue.Queue[YahooDraftPick] = queue.Queue()
            poller = YahooDraftPoller(
                source=source,
                league_key="449.l.12345",
                season=2026,
                interval=5.0,
                pick_queue=pick_queue,
            )

            # Run poll_once from a background thread — would raise
            # ProgrammingError with check_same_thread=True
            errors: list[Exception] = []

            def target() -> None:
                try:
                    poller.poll_once()
                except Exception as exc:
                    errors.append(exc)

            t = threading.Thread(target=target)
            t.start()
            t.join(timeout=5)

            assert not errors, f"Background thread raised: {errors[0]}"
            assert pick_queue.qsize() == 1
        finally:
            pool.close_all()
