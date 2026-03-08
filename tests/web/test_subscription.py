"""Tests for GraphQL subscription event publishing from mutations."""

from fastapi.testclient import TestClient

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.repos import SqliteDraftSessionRepo
from fantasy_baseball_manager.web import SessionManager, create_app
from fantasy_baseball_manager.web.event_bus import EventBus
from fantasy_baseball_manager.web.types import PickEvent, SessionEvent, UndoEvent

from .conftest import _LEAGUE, _seed_data


def _make_client_with_bus() -> tuple:
    """Create a test client with a shared EventBus."""
    conn = create_connection(":memory:", check_same_thread=False)
    provider = SingleConnectionProvider(conn)
    _seed_data(provider)
    container = AnalysisContainer(provider)
    session_repo = SqliteDraftSessionRepo(provider)
    event_bus = EventBus()
    session_manager = SessionManager(
        session_repo=session_repo,
        valuation_repo=container.valuation_repo,
        player_repo=container.player_repo,
        adp_repo=container.adp_repo,
        player_profile_service=container.player_profile_service,
        league=_LEAGUE,
        adp_provider="fantasypros",
    )
    app = create_app(container, _LEAGUE, session_manager=session_manager, event_bus=event_bus)
    client = TestClient(app)
    return client, event_bus


def _start_session(client) -> int:  # noqa: ANN001
    resp = client.post(
        "/graphql",
        json={
            "query": """
                mutation {
                    startSession(season: 2026) {
                        sessionId
                    }
                }
            """
        },
    )
    return resp.json()["data"]["startSession"]["sessionId"]


def test_mutation_pick_publishes_event() -> None:
    client, bus = _make_client_with_bus()
    session_id = _start_session(client)

    # Subscribe before the pick
    q = bus.subscribe(session_id)

    # The start_session event was published before we subscribed, so queue should be empty
    assert q.empty()

    client.post(
        "/graphql",
        json={
            "query": """
                mutation($sid: Int!) {
                    pick(sessionId: $sid, playerId: 1, position: "OF") {
                        pick { playerId }
                    }
                }
            """,
            "variables": {"sid": session_id},
        },
    )

    event = q.get_nowait()
    assert isinstance(event, PickEvent)
    assert event.pick.player_id == 1
    assert event.session_id == session_id


def test_mutation_undo_publishes_event() -> None:
    client, bus = _make_client_with_bus()
    session_id = _start_session(client)

    # Make a pick first
    client.post(
        "/graphql",
        json={
            "query": """
                mutation($sid: Int!) {
                    pick(sessionId: $sid, playerId: 1, position: "OF") {
                        pick { playerId }
                    }
                }
            """,
            "variables": {"sid": session_id},
        },
    )

    q = bus.subscribe(session_id)

    client.post(
        "/graphql",
        json={
            "query": """
                mutation($sid: Int!) {
                    undo(sessionId: $sid) {
                        pick { playerId }
                    }
                }
            """,
            "variables": {"sid": session_id},
        },
    )

    event = q.get_nowait()
    assert isinstance(event, UndoEvent)
    assert event.pick.player_id == 1


def test_mutation_session_events() -> None:
    client, bus = _make_client_with_bus()

    # Subscribe to a session ID we'll create — we need to know the ID ahead of time,
    # so subscribe to session 1 (first session created will be 1)
    q = bus.subscribe(1)

    session_id = _start_session(client)
    assert session_id == 1

    start_event = q.get_nowait()
    assert isinstance(start_event, SessionEvent)
    assert start_event.event_type == "started"

    client.post(
        "/graphql",
        json={
            "query": """
                mutation($sid: Int!) {
                    endSession(sessionId: $sid)
                }
            """,
            "variables": {"sid": session_id},
        },
    )

    end_event = q.get_nowait()
    assert isinstance(end_event, SessionEvent)
    assert end_event.event_type == "ended"


def test_events_published_to_correct_session() -> None:
    client, bus = _make_client_with_bus()
    sid1 = _start_session(client)
    sid2 = _start_session(client)

    q1 = bus.subscribe(sid1)
    q2 = bus.subscribe(sid2)

    client.post(
        "/graphql",
        json={
            "query": """
                mutation($sid: Int!) {
                    pick(sessionId: $sid, playerId: 1, position: "OF") {
                        pick { playerId }
                    }
                }
            """,
            "variables": {"sid": sid1},
        },
    )

    assert not q1.empty()
    assert q2.empty()
