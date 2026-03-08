import asyncio
import contextlib
from typing import Any


class EventBus:
    """Per-session fan-out event bus using asyncio queues."""

    def __init__(self) -> None:
        self._subscribers: dict[int, list[asyncio.Queue[Any]]] = {}

    def subscribe(self, session_id: int) -> asyncio.Queue[Any]:
        q: asyncio.Queue[Any] = asyncio.Queue()
        self._subscribers.setdefault(session_id, []).append(q)
        return q

    def unsubscribe(self, session_id: int, q: asyncio.Queue[Any]) -> None:
        subs = self._subscribers.get(session_id)
        if subs is not None:
            with contextlib.suppress(ValueError):
                subs.remove(q)

    async def publish(self, session_id: int, event: Any) -> None:
        for q in self._subscribers.get(session_id, []):
            q.put_nowait(event)
