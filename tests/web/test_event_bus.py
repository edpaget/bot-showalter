import asyncio
from dataclasses import dataclass

import pytest

from fantasy_baseball_manager.web.event_bus import EventBus


@dataclass
class _FakeEvent:
    value: str


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


def test_subscribe_and_receive(bus: EventBus) -> None:
    q = bus.subscribe(1)
    event = _FakeEvent("hello")
    asyncio.run(bus.publish(1, event))
    received = q.get_nowait()
    assert received == event


def test_multiple_subscribers(bus: EventBus) -> None:
    q1 = bus.subscribe(1)
    q2 = bus.subscribe(1)
    event = _FakeEvent("broadcast")
    asyncio.run(bus.publish(1, event))
    assert q1.get_nowait() == event
    assert q2.get_nowait() == event


def test_unsubscribe(bus: EventBus) -> None:
    q = bus.subscribe(1)
    bus.unsubscribe(1, q)
    asyncio.run(bus.publish(1, _FakeEvent("missed")))
    assert q.empty()


def test_publish_no_subscribers(bus: EventBus) -> None:
    asyncio.run(bus.publish(999, _FakeEvent("no-one-listening")))


def test_separate_sessions(bus: EventBus) -> None:
    q1 = bus.subscribe(1)
    q2 = bus.subscribe(2)
    asyncio.run(bus.publish(1, _FakeEvent("session-1")))
    assert not q1.empty()
    assert q2.empty()
