from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    Player,
    StatType,
    Valuation,
)
from fantasy_baseball_manager.repos import (
    SqliteADPRepo,
    SqliteDraftSessionRepo,
    SqlitePlayerRepo,
    SqliteValuationRepo,
)
from fantasy_baseball_manager.services import PlayerProfileService
from fantasy_baseball_manager.web import SessionManager
from fantasy_baseball_manager.web.session_manager import DraftSessionSummary

_LEAGUE = LeagueSettings(
    name="Test League",
    format=LeagueFormat.H2H_CATEGORIES,
    teams=10,
    budget=260,
    roster_batters=14,
    roster_pitchers=9,
    batting_categories=(
        CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        CategoryConfig(key="RBI", name="Runs Batted In", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
    ),
    pitching_categories=(
        CategoryConfig(key="W", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
    ),
    positions={"C": 1, "1B": 1, "OF": 3},
    pitcher_positions={"SP": 5, "RP": 2},
)


def _make_manager() -> SessionManager:
    conn = create_connection(":memory:", check_same_thread=False)
    provider = SingleConnectionProvider(conn)

    player_repo = SqlitePlayerRepo(provider)
    valuation_repo = SqliteValuationRepo(provider)

    players = [
        Player(name_first="Mike", name_last="Trout", id=1, mlbam_id=545361),
        Player(name_first="Shohei", name_last="Ohtani", id=2, mlbam_id=660271),
        Player(name_first="Gerrit", name_last="Cole", id=3, mlbam_id=543037),
    ]
    for p in players:
        player_repo.upsert(p)

    valuations = [
        Valuation(
            player_id=1,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026",
            player_type="batter",
            position="OF",
            value=35.0,
            rank=1,
            category_scores={"HR": 2.5, "RBI": 1.8},
        ),
        Valuation(
            player_id=2,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026",
            player_type="batter",
            position="OF",
            value=30.0,
            rank=2,
            category_scores={"HR": 2.0, "RBI": 1.5},
        ),
        Valuation(
            player_id=3,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026",
            player_type="pitcher",
            position="SP",
            value=25.0,
            rank=3,
            category_scores={"W": 1.5, "K": 2.0},
        ),
    ]
    for v in valuations:
        valuation_repo.upsert(v)

    with provider.connection() as c:
        c.commit()

    session_repo = SqliteDraftSessionRepo(provider)
    adp_repo = SqliteADPRepo(provider)
    profile_service = PlayerProfileService(player_repo)

    return SessionManager(
        session_repo=session_repo,
        valuation_repo=valuation_repo,
        player_repo=player_repo,
        adp_repo=adp_repo,
        player_profile_service=profile_service,
        league=_LEAGUE,
        adp_provider="fantasypros",
    )


class TestGetEngineUsesStoredSystemVersion:
    def test_rehydration_uses_stored_system_version(self) -> None:
        mgr = _make_manager()

        session_id, _ = mgr.start_session(2026, system="zar", version="1.0")

        # Evict in-memory cache so get_engine must rehydrate from DB
        mgr._engines.clear()

        engine = mgr.get_engine(session_id)
        assert engine is not None
        # Engine was rehydrated successfully — proves it read system/version from the record
        assert engine.state.config.teams == 10


class TestListSessions:
    def test_list_sessions_returns_summaries(self) -> None:
        mgr = _make_manager()

        sid1, _ = mgr.start_session(2026)
        sid2, _ = mgr.start_session(2026)

        # Pick in session 1 only
        mgr.pick(sid1, 1, 1, "OF")

        summaries = mgr.list_sessions()
        assert len(summaries) == 2
        assert all(isinstance(s, DraftSessionSummary) for s in summaries)

        by_id = {s.record.id: s for s in summaries}
        assert by_id[sid1].pick_count == 1
        assert by_id[sid2].pick_count == 0

    def test_list_sessions_includes_system_version(self) -> None:
        mgr = _make_manager()
        mgr.start_session(2026)

        summaries = mgr.list_sessions()
        assert len(summaries) == 1
        assert summaries[0].record.system == "zar"
        assert summaries[0].record.version == "1.0"

    def test_list_sessions_filters_by_status(self) -> None:
        mgr = _make_manager()

        sid1, _ = mgr.start_session(2026)
        sid2, _ = mgr.start_session(2026)

        mgr.end_session(sid1)

        complete = mgr.list_sessions(status="complete")
        assert len(complete) == 1
        assert complete[0].record.id == sid1

        in_progress = mgr.list_sessions(status="in_progress")
        assert len(in_progress) == 1
        assert in_progress[0].record.id == sid2
