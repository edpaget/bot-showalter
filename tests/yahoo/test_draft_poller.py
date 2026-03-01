import queue

from fantasy_baseball_manager.domain.yahoo_draft_pick import YahooDraftPick
from fantasy_baseball_manager.yahoo.draft_poller import YahooDraftPoller


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
            source=source,  # type: ignore[arg-type]
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
            source=source,  # type: ignore[arg-type]
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
            source=source,  # type: ignore[arg-type]
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
            source=source,  # type: ignore[arg-type]
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
