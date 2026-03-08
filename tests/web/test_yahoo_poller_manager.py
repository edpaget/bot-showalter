"""Tests for YahooPollerManager — Yahoo draft polling lifecycle and event bridging."""

import asyncio

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import YahooDraftPick, YahooTeam
from fantasy_baseball_manager.repos import SqliteDraftSessionRepo
from fantasy_baseball_manager.web.event_bus import EventBus
from fantasy_baseball_manager.web.session_manager import SessionManager
from fantasy_baseball_manager.web.types import PickEvent
from fantasy_baseball_manager.web.yahoo_poller_manager import YahooPollerManager, YahooPollStatus

from .conftest import _LEAGUE, _seed_data


class FakeDraftSource:
    """Fake YahooDraftSourceProto that returns canned picks."""

    def __init__(self, picks: list[YahooDraftPick] | None = None) -> None:
        self._picks = picks or []

    def fetch_draft_results(self, league_key: str, season: int) -> list[YahooDraftPick]:
        return list(self._picks)

    def set_picks(self, picks: list[YahooDraftPick]) -> None:
        self._picks = picks


class FakeTeamRepo:
    """Fake YahooTeamRepo."""

    def __init__(self, teams: list[YahooTeam] | None = None) -> None:
        self._teams = teams or []

    def upsert(self, team: YahooTeam) -> int:
        return 1

    def get_by_league_key(self, league_key: str) -> list[YahooTeam]:
        return list(self._teams)

    def get_user_team(self, league_key: str) -> YahooTeam | None:
        return None


def _make_deps() -> tuple[SessionManager, EventBus, SingleConnectionProvider]:
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
    return session_manager, event_bus, provider


def _make_yahoo_pick(
    *,
    player_id: int,
    player_name: str,
    position: str,
    team_key: str = "422.l.12345.t.1",
    cost: int | None = None,
) -> YahooDraftPick:
    return YahooDraftPick(
        league_key="422.l.12345",
        season=2026,
        round=1,
        pick=1,
        team_key=team_key,
        yahoo_player_key="422.p.1",
        player_id=player_id,
        player_name=player_name,
        position=position,
        cost=cost,
    )


def test_start_and_stop_polling() -> None:
    session_manager, event_bus, _ = _make_deps()
    session_id, _ = session_manager.start_session(2026)

    fake_source = FakeDraftSource()
    fake_teams = FakeTeamRepo(
        teams=[
            YahooTeam(
                team_key="422.l.12345.t.1",
                league_key="422.l.12345",
                team_id=1,
                name="Team 1",
                manager_name="Mgr",
                is_owned_by_user=True,
            )
        ]
    )

    mgr = YahooPollerManager(
        _draft_source=fake_source,
        _session_manager=session_manager,
        _event_bus=event_bus,
        _team_repo=fake_teams,
        _player_repo=session_manager._player_repo,
    )

    result = asyncio.run(mgr.start_polling(session_id, "422.l.12345"))
    assert result is True

    status = mgr.get_status(session_id)
    assert status.active is True

    # Starting again should return False
    result2 = asyncio.run(mgr.start_polling(session_id, "422.l.12345"))
    assert result2 is False

    result3 = asyncio.run(mgr.stop_polling(session_id))
    assert result3 is True

    status = mgr.get_status(session_id)
    assert status.active is False


def test_stop_nonexistent_session() -> None:
    session_manager, event_bus, _ = _make_deps()

    mgr = YahooPollerManager(
        _draft_source=FakeDraftSource(),
        _session_manager=session_manager,
        _event_bus=event_bus,
        _team_repo=FakeTeamRepo(),
        _player_repo=session_manager._player_repo,
    )

    result = asyncio.run(mgr.stop_polling(999))
    assert result is False


def test_poll_status_default() -> None:
    session_manager, event_bus, _ = _make_deps()

    mgr = YahooPollerManager(
        _draft_source=FakeDraftSource(),
        _session_manager=session_manager,
        _event_bus=event_bus,
        _team_repo=FakeTeamRepo(),
        _player_repo=session_manager._player_repo,
    )

    status = mgr.get_status(999)
    assert status == YahooPollStatus()
    assert status.active is False
    assert status.picks_ingested == 0


def test_shutdown_stops_all() -> None:
    session_manager, event_bus, _ = _make_deps()
    sid1, _ = session_manager.start_session(2026)
    sid2, _ = session_manager.start_session(2026)

    fake_teams = FakeTeamRepo(
        teams=[
            YahooTeam(
                team_key="422.l.12345.t.1",
                league_key="422.l.12345",
                team_id=1,
                name="Team 1",
                manager_name="Mgr",
                is_owned_by_user=True,
            )
        ]
    )

    mgr = YahooPollerManager(
        _draft_source=FakeDraftSource(),
        _session_manager=session_manager,
        _event_bus=event_bus,
        _team_repo=fake_teams,
        _player_repo=session_manager._player_repo,
    )

    asyncio.run(mgr.start_polling(sid1, "422.l.12345"))
    asyncio.run(mgr.start_polling(sid2, "422.l.12345"))

    assert mgr.get_status(sid1).active is True
    assert mgr.get_status(sid2).active is True

    asyncio.run(mgr.shutdown())

    assert mgr.get_status(sid1).active is False
    assert mgr.get_status(sid2).active is False


def test_yahoo_picks_bridge_to_events() -> None:
    """Feed picks through fake source into the bridge loop and verify PickEvents on the bus."""
    session_manager, event_bus, _ = _make_deps()
    session_id, _ = session_manager.start_session(2026)

    yahoo_pick = _make_yahoo_pick(
        player_id=1,
        player_name="Mike Trout",
        position="OF",
        team_key="422.l.12345.t.1",
    )

    fake_source = FakeDraftSource(picks=[yahoo_pick])
    fake_teams = FakeTeamRepo(
        teams=[
            YahooTeam(
                team_key="422.l.12345.t.1",
                league_key="422.l.12345",
                team_id=1,
                name="Team 1",
                manager_name="Mgr",
                is_owned_by_user=True,
            ),
        ]
    )

    mgr = YahooPollerManager(
        _draft_source=fake_source,
        _session_manager=session_manager,
        _event_bus=event_bus,
        _team_repo=fake_teams,
        _player_repo=session_manager._player_repo,
    )

    # Subscribe to events
    q = event_bus.subscribe(session_id)

    # Directly test the bridge logic by simulating what _bridge_loop does:
    # Put a pick in the thread queue and run the bridge loop briefly
    async def _run_bridge_test() -> None:
        # Start polling (creates the poller and bridge task)
        await mgr.start_polling(session_id, "422.l.12345")

        # Put the yahoo pick directly on the thread queue
        mgr._thread_queues[session_id].put(yahoo_pick)

        # Give the bridge loop time to process
        await asyncio.sleep(0.5)

        await mgr.stop_polling(session_id)

    asyncio.run(_run_bridge_test())

    # Verify PickEvent was published
    assert not q.empty()
    event = q.get_nowait()
    assert isinstance(event, PickEvent)
    assert event.pick.player_id == 1
    assert event.pick.team == 1
    assert event.session_id == session_id

    # Verify pick was persisted
    engine = session_manager.get_engine(session_id)
    assert len(engine.state.picks) == 1
    assert engine.state.picks[0].player_id == 1

    # Verify status was updated
    assert mgr.get_status(session_id).picks_ingested == 1
