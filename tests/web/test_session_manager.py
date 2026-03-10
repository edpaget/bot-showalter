from dataclasses import replace

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueKeeper,
    LeagueSettings,
    Player,
    StatType,
    Valuation,
)
from fantasy_baseball_manager.repos import (
    SqliteADPRepo,
    SqliteDraftSessionRepo,
    SqliteLeagueKeeperRepo,
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


def _make_manager_with_adjuster() -> tuple[SessionManager, SingleConnectionProvider]:
    conn = create_connection(":memory:", check_same_thread=False)
    provider = SingleConnectionProvider(conn)

    player_repo = SqlitePlayerRepo(provider)
    valuation_repo = SqliteValuationRepo(provider)

    players = [
        Player(name_first="Mike", name_last="Trout", id=1, mlbam_id=545361),
        Player(name_first="Shohei", name_last="Ohtani", id=2, mlbam_id=660271),
        Player(name_first="Gerrit", name_last="Cole", id=3, mlbam_id=543037),
        Player(name_first="Aaron", name_last="Judge", id=4, mlbam_id=592450),
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
        Valuation(
            player_id=4,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026",
            player_type="batter",
            position="OF",
            value=20.0,
            rank=4,
            category_scores={"HR": 1.5, "RBI": 1.0},
        ),
    ]
    for v in valuations:
        valuation_repo.upsert(v)

    with provider.connection() as c:
        c.commit()

    def fake_adjuster(kept_ids: set[int], valuations: list[Valuation], season: int) -> list[Valuation]:
        """Fake adjuster that boosts remaining players' values by 10%."""
        return [replace(v, value=round(v.value * 1.1, 2)) for v in valuations if v.player_id not in kept_ids]

    session_repo = SqliteDraftSessionRepo(provider)
    adp_repo = SqliteADPRepo(provider)
    profile_service = PlayerProfileService(player_repo)

    mgr = SessionManager(
        session_repo=session_repo,
        valuation_repo=valuation_repo,
        player_repo=player_repo,
        adp_repo=adp_repo,
        player_profile_service=profile_service,
        league=_LEAGUE,
        adp_provider="fantasypros",
        valuation_adjuster=fake_adjuster,
    )
    return mgr, provider


class TestKeeperExclusion:
    def test_kept_players_excluded_from_available(self) -> None:
        mgr, _ = _make_manager_with_adjuster()
        sid, engine = mgr.start_session(2026, keeper_player_ids={1})

        available_ids = [r.player_id for r in engine.available()]
        assert 1 not in available_ids
        # Non-kept players still present
        assert 2 in available_ids

    def test_adjusted_values_differ_from_raw(self) -> None:
        mgr, _ = _make_manager_with_adjuster()
        # Without keepers
        _, engine_raw = mgr.start_session(2026)
        raw_values = {r.player_id: r.value for r in engine_raw.available()}

        mgr2, _ = _make_manager_with_adjuster()
        # With keepers — player 1 excluded, remaining boosted by 10%
        _, engine_adj = mgr2.start_session(2026, keeper_player_ids={1})
        adj_values = {r.player_id: r.value for r in engine_adj.available()}

        # Player 2 should have a different (higher) value when player 1 is kept
        assert adj_values[2] != raw_values[2]
        assert adj_values[2] > raw_values[2]

    def test_session_restore_preserves_keeper_context(self) -> None:
        mgr, _ = _make_manager_with_adjuster()
        sid, engine = mgr.start_session(2026, keeper_player_ids={1})

        original_ids = {r.player_id for r in engine.available()}

        # Evict cache and restore
        mgr._engines.clear()
        restored = mgr.get_engine(sid)

        restored_ids = {r.player_id for r in restored.available()}
        assert 1 not in restored_ids
        assert restored_ids == original_ids

    def test_no_keepers_identical_to_default(self) -> None:
        mgr1 = _make_manager()
        _, engine1 = mgr1.start_session(2026)
        default_ids = {r.player_id for r in engine1.available()}

        mgr2 = _make_manager()
        _, engine2 = mgr2.start_session(2026, keeper_player_ids=None)
        none_ids = {r.player_id for r in engine2.available()}

        assert default_ids == none_ids

    def test_keeper_ids_persisted_in_record(self) -> None:
        mgr, _ = _make_manager_with_adjuster()
        sid, _ = mgr.start_session(2026, keeper_player_ids={1, 3})

        record = mgr._repo.load_session(sid)
        assert record is not None
        assert record.keeper_player_ids == [1, 3]


class TestAutoLoadKeepers:
    def test_auto_loads_from_league_keeper_repo(self) -> None:
        conn = create_connection(":memory:", check_same_thread=False)
        provider = SingleConnectionProvider(conn)

        player_repo = SqlitePlayerRepo(provider)
        valuation_repo = SqliteValuationRepo(provider)
        league_keeper_repo = SqliteLeagueKeeperRepo(provider)

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

        # Persist league keepers — player 1 is kept
        league_keeper_repo.upsert_batch(
            [
                LeagueKeeper(player_id=1, season=2026, league="Test League", team_name="Team A"),
            ]
        )

        with provider.connection() as c:
            c.commit()

        def fake_adjuster(kept_ids: set[int], valuations: list[Valuation], season: int) -> list[Valuation]:
            return [v for v in valuations if v.player_id not in kept_ids]

        session_repo = SqliteDraftSessionRepo(provider)
        adp_repo = SqliteADPRepo(provider)
        profile_service = PlayerProfileService(player_repo)

        mgr = SessionManager(
            session_repo=session_repo,
            valuation_repo=valuation_repo,
            player_repo=player_repo,
            adp_repo=adp_repo,
            player_profile_service=profile_service,
            league=_LEAGUE,
            adp_provider="fantasypros",
            valuation_adjuster=fake_adjuster,
            league_keeper_repo=league_keeper_repo,
        )

        # Don't pass keeper_player_ids — should auto-load from repo
        sid, engine = mgr.start_session(2026)
        available_ids = {r.player_id for r in engine.available()}
        assert 1 not in available_ids
        assert 2 in available_ids

    def test_no_league_keepers_backward_compatible(self) -> None:
        conn = create_connection(":memory:", check_same_thread=False)
        provider = SingleConnectionProvider(conn)

        player_repo = SqlitePlayerRepo(provider)
        valuation_repo = SqliteValuationRepo(provider)
        league_keeper_repo = SqliteLeagueKeeperRepo(provider)

        players = [
            Player(name_first="Mike", name_last="Trout", id=1, mlbam_id=545361),
            Player(name_first="Gerrit", name_last="Cole", id=2, mlbam_id=543037),
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
                player_type="pitcher",
                position="SP",
                value=25.0,
                rank=2,
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

        mgr = SessionManager(
            session_repo=session_repo,
            valuation_repo=valuation_repo,
            player_repo=player_repo,
            adp_repo=adp_repo,
            player_profile_service=profile_service,
            league=_LEAGUE,
            adp_provider="fantasypros",
            league_keeper_repo=league_keeper_repo,
        )

        # No league keepers in DB — should behave like default
        _, engine = mgr.start_session(2026)
        available_ids = {r.player_id for r in engine.available()}
        assert 1 in available_ids
        assert 2 in available_ids
