import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import queue

    from fantasy_baseball_manager.domain import YahooDraftPick
    from fantasy_baseball_manager.repos import YahooDraftSourceProto

logger = logging.getLogger(__name__)


class YahooDraftPoller:
    def __init__(
        self,
        source: YahooDraftSourceProto,
        league_key: str,
        season: int,
        interval: float,
        pick_queue: queue.Queue[YahooDraftPick],
    ) -> None:
        self._source = source
        self._league_key = league_key
        self._season = season
        self._interval = interval
        self._pick_queue = pick_queue
        self._known_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> list[YahooDraftPick]:
        picks = self._source.fetch_draft_results(self._league_key, self._season)
        if len(picks) <= self._known_count:
            return []

        new_picks = picks[self._known_count :]
        self._known_count = len(picks)

        for pick in new_picks:
            self._pick_queue.put(pick)

        return new_picks

    def start(self) -> None:  # pragma: no cover
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:  # pragma: no cover
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1)

    def _run(self) -> None:  # pragma: no cover
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.exception("Error polling Yahoo draft results")
            self._stop_event.wait(self._interval)
