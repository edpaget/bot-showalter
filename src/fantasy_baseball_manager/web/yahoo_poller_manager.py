import asyncio
import contextlib
import functools
import logging
import queue
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.services import (
    build_player_id_aliases,
    build_team_map,
    ingest_yahoo_pick,
)
from fantasy_baseball_manager.web.types import DraftPickType, PickEvent
from fantasy_baseball_manager.yahoo.draft_poller import YahooDraftPoller

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import YahooDraftPick
    from fantasy_baseball_manager.repos import YahooDraftSourceProto, YahooTeamRepo
    from fantasy_baseball_manager.web.event_bus import EventBus
    from fantasy_baseball_manager.web.session_manager import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class YahooPollStatus:
    active: bool = False
    last_poll_at: str | None = None
    picks_ingested: int = 0


@dataclass
class YahooPollerManager:
    _draft_source: YahooDraftSourceProto
    _session_manager: SessionManager
    _event_bus: EventBus
    _team_repo: YahooTeamRepo
    _pollers: dict[int, YahooDraftPoller] = field(default_factory=dict)
    _bridge_tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict)
    _thread_queues: dict[int, queue.Queue[YahooDraftPick]] = field(default_factory=dict)
    _status: dict[int, YahooPollStatus] = field(default_factory=dict)

    async def start_polling(
        self,
        session_id: int,
        league_key: str,
        *,
        interval: float = 5.0,
    ) -> bool:
        if session_id in self._pollers:
            return False

        teams = self._team_repo.get_by_league_key(league_key)
        team_map = build_team_map(teams)

        engine = self._session_manager.get_engine(session_id)

        # Build player name map for alias resolution (include all players, not just available)
        board_names: dict[int, str] = {}
        for row in engine.state.available_pool.values():
            board_names[row.player_id] = row.player_name
        for pick in engine.state.picks:
            board_names[pick.player_id] = pick.player_name

        # Get initial picks from source for alias building
        initial_picks = self._draft_source.fetch_draft_results(league_key, engine.state.config.season)
        id_aliases = build_player_id_aliases(initial_picks, board_names)

        pick_queue: queue.Queue[YahooDraftPick] = queue.Queue()
        self._thread_queues[session_id] = pick_queue

        poller = YahooDraftPoller(
            source=self._draft_source,
            league_key=league_key,
            season=engine.state.config.season,
            interval=interval,
            pick_queue=pick_queue,
        )
        poller.start()
        self._pollers[session_id] = poller
        self._status[session_id] = YahooPollStatus(active=True)

        task = asyncio.create_task(
            self._bridge_loop(session_id, team_map, id_aliases),
        )
        self._bridge_tasks[session_id] = task

        return True

    async def stop_polling(self, session_id: int) -> bool:
        if session_id not in self._pollers:
            return False

        self._pollers[session_id].stop()
        del self._pollers[session_id]

        task = self._bridge_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._thread_queues.pop(session_id, None)

        status = self._status.get(session_id)
        if status is not None:
            status.active = False

        return True

    def get_status(self, session_id: int) -> YahooPollStatus:
        return self._status.get(session_id, YahooPollStatus())

    async def shutdown(self) -> None:
        for session_id in list(self._pollers):
            await self.stop_polling(session_id)

    async def _bridge_loop(
        self,
        session_id: int,
        team_map: dict[str, int],
        id_aliases: dict[int, int],
    ) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                yahoo_pick = await loop.run_in_executor(
                    None,
                    functools.partial(self._thread_queues[session_id].get, timeout=1.0),
                )
            except queue.Empty:
                continue

            engine = self._session_manager.get_engine(session_id)
            draft_pick = ingest_yahoo_pick(
                engine.pick,
                set(engine.state.available_pool),
                yahoo_pick,
                team_map,
                id_aliases=id_aliases,
                roster_slots=engine.state.config.roster_slots,
                team_rosters=engine.state.team_rosters,
            )
            if draft_pick is not None:
                self._session_manager.persist_external_pick(session_id, draft_pick)
                status = self._status.get(session_id)
                if status is not None:
                    status.picks_ingested += 1
                await self._event_bus.publish(
                    session_id,
                    PickEvent(
                        pick=DraftPickType.from_domain(draft_pick),
                        session_id=session_id,
                    ),
                )
