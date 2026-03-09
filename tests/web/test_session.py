"""Tests for GraphQL session mutations and queries (phase 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.repos import SqliteDraftSessionRepo
from fantasy_baseball_manager.web import SessionManager, create_app

from .conftest import _LEAGUE

if TYPE_CHECKING:
    from fantasy_baseball_manager.db.pool import SingleConnectionProvider

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_START_MUTATION = """
mutation StartSession($season: Int!) {
    startSession(season: $season, teams: 10, userTeam: 1, format: "snake") {
        sessionId
        currentPick
        picks { pickNumber }
        format
        teams
        userTeam
    }
}
"""

_PICK_MUTATION = """
mutation Pick($sessionId: Int!, $playerId: Int!, $position: Position!) {
    pick(sessionId: $sessionId, playerId: $playerId, position: $position) {
        pick { pickNumber playerName position team }
        state { sessionId currentPick }
        recommendations { playerName }
        roster { playerName position }
        needs { position remaining }
    }
}
"""

_UNDO_MUTATION = """
mutation Undo($sessionId: Int!) {
    undo(sessionId: $sessionId) {
        pick { pickNumber playerName }
        state { sessionId currentPick }
        recommendations { playerName }
        roster { playerName }
        needs { position remaining }
    }
}
"""

_END_MUTATION = """
mutation EndSession($sessionId: Int!) {
    endSession(sessionId: $sessionId)
}
"""


def _gql(client: TestClient, query: str, variables: dict | None = None) -> dict:
    resp = client.post("/graphql", json={"query": query, "variables": variables or {}})
    assert resp.status_code == 200
    return resp.json()


def _start_session(client: TestClient) -> int:
    result = _gql(client, _START_MUTATION, {"season": 2026})
    assert "errors" not in result, result.get("errors")
    return result["data"]["startSession"]["sessionId"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_start_session(session_client: TestClient) -> None:
    result = _gql(session_client, _START_MUTATION, {"season": 2026})
    assert "errors" not in result, result.get("errors")
    data = result["data"]["startSession"]
    assert data["sessionId"] >= 1
    assert data["currentPick"] == 1
    assert data["picks"] == []
    assert data["format"] == "snake"
    assert data["teams"] == 10
    assert data["userTeam"] == 1


def test_pick_and_state(session_client: TestClient) -> None:
    sid = _start_session(session_client)

    result = _gql(
        session_client,
        _PICK_MUTATION,
        {
            "sessionId": sid,
            "playerId": 1,
            "position": "OF",
        },
    )
    assert "errors" not in result, result.get("errors")
    data = result["data"]["pick"]
    assert data["pick"]["pickNumber"] == 1
    assert data["pick"]["playerName"] == "Mike Trout"
    assert data["state"]["currentPick"] == 2

    # Query available — pool should have shrunk by 1
    avail_result = _gql(
        session_client,
        """
        query Available($sid: Int!) {
            available(sessionId: $sid) { playerId }
        }
    """,
        {"sid": sid},
    )
    available_ids = [r["playerId"] for r in avail_result["data"]["available"]]
    assert 1 not in available_ids


def test_pick_returns_result_bundle(session_client: TestClient) -> None:
    sid = _start_session(session_client)
    result = _gql(
        session_client,
        _PICK_MUTATION,
        {
            "sessionId": sid,
            "playerId": 1,
            "position": "OF",
        },
    )
    assert "errors" not in result, result.get("errors")
    data = result["data"]["pick"]
    assert "pick" in data
    assert "state" in data
    assert "recommendations" in data
    assert "roster" in data
    assert "needs" in data


def test_undo(session_client: TestClient) -> None:
    sid = _start_session(session_client)
    # Pick then undo
    _gql(
        session_client,
        _PICK_MUTATION,
        {
            "sessionId": sid,
            "playerId": 1,
            "position": "OF",
        },
    )
    result = _gql(session_client, _UNDO_MUTATION, {"sessionId": sid})
    assert "errors" not in result, result.get("errors")
    data = result["data"]["undo"]
    assert data["pick"]["playerName"] == "Mike Trout"
    assert data["state"]["currentPick"] == 1

    # Player should be back in pool
    avail_result = _gql(
        session_client,
        """
        query Available($sid: Int!) {
            available(sessionId: $sid) { playerId }
        }
    """,
        {"sid": sid},
    )
    available_ids = [r["playerId"] for r in avail_result["data"]["available"]]
    assert 1 in available_ids


def test_session_hydration(session_provider: SingleConnectionProvider) -> None:
    """Start session, pick, then create a fresh client (simulating restart) and verify hydration."""
    container = AnalysisContainer(session_provider)
    session_repo = SqliteDraftSessionRepo(session_provider)
    mgr = SessionManager(
        session_repo=session_repo,
        valuation_repo=container.valuation_repo,
        player_repo=container.player_repo,
        adp_repo=container.adp_repo,
        player_profile_service=container.player_profile_service,
        league=_LEAGUE,
        adp_provider="fantasypros",
    )
    app1 = create_app(container, _LEAGUE, session_manager=mgr, default_system="zar", default_version="1.0")
    client1 = TestClient(app1)

    # Start and pick
    sid = _start_session(client1)
    _gql(client1, _PICK_MUTATION, {"sessionId": sid, "playerId": 1, "position": "OF"})

    # Create fresh SessionManager (simulates server restart — no cached engines)
    mgr2 = SessionManager(
        session_repo=session_repo,
        valuation_repo=container.valuation_repo,
        player_repo=container.player_repo,
        adp_repo=container.adp_repo,
        player_profile_service=container.player_profile_service,
        league=_LEAGUE,
        adp_provider="fantasypros",
    )
    app2 = create_app(container, _LEAGUE, session_manager=mgr2, default_system="zar", default_version="1.0")
    client2 = TestClient(app2)

    # Query session — should hydrate from DB
    result = _gql(
        client2,
        """
        query Session($sid: Int!) {
            session(sessionId: $sid) {
                sessionId
                currentPick
                picks { playerName }
            }
        }
    """,
        {"sid": sid},
    )
    assert "errors" not in result, result.get("errors")
    data = result["data"]["session"]
    assert data["currentPick"] == 2
    assert len(data["picks"]) == 1
    assert data["picks"][0]["playerName"] == "Mike Trout"


def test_sessions_list(session_client: TestClient) -> None:
    sid1 = _start_session(session_client)
    sid2 = _start_session(session_client)

    # Pick in session 1
    _gql(
        session_client,
        _PICK_MUTATION,
        {
            "sessionId": sid1,
            "playerId": 1,
            "position": "OF",
        },
    )

    result = _gql(
        session_client,
        """
        query Sessions {
            sessions { id pickCount status }
        }
    """,
    )
    assert "errors" not in result, result.get("errors")
    sessions = result["data"]["sessions"]
    assert len(sessions) >= 2
    by_id = {s["id"]: s for s in sessions}
    assert by_id[sid1]["pickCount"] == 1
    assert by_id[sid2]["pickCount"] == 0


def test_end_session(session_client: TestClient) -> None:
    sid = _start_session(session_client)
    result = _gql(session_client, _END_MUTATION, {"sessionId": sid})
    assert "errors" not in result, result.get("errors")
    assert result["data"]["endSession"] is True

    # Verify status is complete
    list_result = _gql(
        session_client,
        """
        query Sessions {
            sessions(status: "complete") { id status }
        }
    """,
    )
    sessions = list_result["data"]["sessions"]
    ids = [s["id"] for s in sessions]
    assert sid in ids


def test_recommendations(session_client: TestClient) -> None:
    sid = _start_session(session_client)
    result = _gql(
        session_client,
        """
        query Recs($sid: Int!) {
            recommendations(sessionId: $sid) { playerName position value score reason }
        }
    """,
        {"sid": sid},
    )
    assert "errors" not in result, result.get("errors")
    recs = result["data"]["recommendations"]
    assert len(recs) > 0
    assert "playerName" in recs[0]


def test_roster_query(session_client: TestClient) -> None:
    sid = _start_session(session_client)
    _gql(
        session_client,
        _PICK_MUTATION,
        {
            "sessionId": sid,
            "playerId": 1,
            "position": "OF",
        },
    )
    result = _gql(
        session_client,
        """
        query Roster($sid: Int!) {
            roster(sessionId: $sid) { playerName position }
        }
    """,
        {"sid": sid},
    )
    assert "errors" not in result, result.get("errors")
    roster = result["data"]["roster"]
    assert len(roster) == 1
    assert roster[0]["playerName"] == "Mike Trout"


def test_needs_query(session_client: TestClient) -> None:
    sid = _start_session(session_client)
    result = _gql(
        session_client,
        """
        query Needs($sid: Int!) {
            needs(sessionId: $sid) { position remaining }
        }
    """,
        {"sid": sid},
    )
    assert "errors" not in result, result.get("errors")
    needs = result["data"]["needs"]
    assert len(needs) > 0
    positions = {n["position"] for n in needs}
    assert "OF" in positions
    assert "SP" in positions


def test_available_query(session_client: TestClient) -> None:
    sid = _start_session(session_client)
    result = _gql(
        session_client,
        """
        query Available($sid: Int!, $pos: Position, $limit: Int!) {
            available(sessionId: $sid, position: $pos, limit: $limit) {
                playerId playerName position
            }
        }
    """,
        {"sid": sid, "pos": "OF", "limit": 2},
    )
    assert "errors" not in result, result.get("errors")
    available = result["data"]["available"]
    assert len(available) <= 2
    for row in available:
        assert row["position"] == "OF"


def test_pick_invalid_player(session_client: TestClient) -> None:
    sid = _start_session(session_client)
    result = _gql(
        session_client,
        _PICK_MUTATION,
        {
            "sessionId": sid,
            "playerId": 99999,
            "position": "OF",
        },
    )
    assert "errors" in result
