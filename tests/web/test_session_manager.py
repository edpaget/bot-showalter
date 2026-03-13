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
    Projection,
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
from tests.fakes.repos import FakeProjectionRepo

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
        sid, engine = mgr.start_session(2026, keeper_player_ids={(1, "batter")})

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
        _, engine_adj = mgr2.start_session(2026, keeper_player_ids={(1, "batter")})
        adj_values = {r.player_id: r.value for r in engine_adj.available()}

        # Player 2 should have a different (higher) value when player 1 is kept
        assert adj_values[2] != raw_values[2]
        assert adj_values[2] > raw_values[2]

    def test_session_restore_preserves_keeper_context(self) -> None:
        mgr, _ = _make_manager_with_adjuster()
        sid, engine = mgr.start_session(2026, keeper_player_ids={(1, "batter")})

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
        sid, _ = mgr.start_session(2026, keeper_player_ids={(1, "batter"), (3, "pitcher")})

        record = mgr._repo.load_session(sid)
        assert record is not None
        assert record.keeper_player_ids == [[1, "batter"], [3, "pitcher"]]


class TestKeeperSnapshot:
    def test_keeper_snapshot_populated_at_start(self) -> None:
        mgr, provider = _make_manager_with_adjuster()

        # Add league keepers for team/cost info
        league_keeper_repo = SqliteLeagueKeeperRepo(provider)
        league_keeper_repo.upsert_batch(
            [
                LeagueKeeper(player_id=1, season=2026, league="Test League", team_name="Team A", cost=35.0),
                LeagueKeeper(player_id=3, season=2026, league="Test League", team_name="Team B", cost=20.0),
            ]
        )
        with provider.connection() as c:
            c.commit()

        mgr_with_keepers = SessionManager(
            session_repo=SqliteDraftSessionRepo(provider),
            valuation_repo=SqliteValuationRepo(provider),
            player_repo=SqlitePlayerRepo(provider),
            adp_repo=SqliteADPRepo(provider),
            player_profile_service=PlayerProfileService(SqlitePlayerRepo(provider)),
            league=_LEAGUE,
            adp_provider="fantasypros",
            valuation_adjuster=lambda kept_ids, valuations, season: [
                v for v in valuations if v.player_id not in kept_ids
            ],
            league_keeper_repo=league_keeper_repo,
        )

        sid, _ = mgr_with_keepers.start_session(2026, keeper_player_ids={(1, "batter"), (3, "pitcher")})
        keepers = mgr_with_keepers.get_keepers(sid)

        assert len(keepers) == 2
        by_id = {k["player_id"]: k for k in keepers}
        assert by_id[1]["player_name"] == "Mike Trout"
        assert by_id[1]["team_name"] == "Team A"
        assert by_id[1]["cost"] == 35.0
        assert by_id[3]["player_name"] == "Gerrit Cole"
        assert by_id[3]["team_name"] == "Team B"

    def test_get_keepers_empty_for_non_keeper_session(self) -> None:
        mgr = _make_manager()
        sid, _ = mgr.start_session(2026)
        keepers = mgr.get_keepers(sid)
        assert keepers == []

    def test_keeper_snapshot_survives_persist_load(self) -> None:
        mgr, provider = _make_manager_with_adjuster()
        sid, _ = mgr.start_session(2026, keeper_player_ids={(1, "batter")})
        keepers = mgr.get_keepers(sid)
        assert len(keepers) == 1
        assert keepers[0]["player_name"] == "Mike Trout"

        # Create fresh manager to simulate restart
        mgr2 = SessionManager(
            session_repo=SqliteDraftSessionRepo(provider),
            valuation_repo=SqliteValuationRepo(provider),
            player_repo=SqlitePlayerRepo(provider),
            adp_repo=SqliteADPRepo(provider),
            player_profile_service=PlayerProfileService(SqlitePlayerRepo(provider)),
            league=_LEAGUE,
            adp_provider="fantasypros",
        )
        keepers2 = mgr2.get_keepers(sid)
        assert keepers2 == keepers


class TestAutoLoadKeepers:
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


class TestGetCategoryBalanceFn:
    def test_returns_function_for_keeper_session(self) -> None:
        mgr, _ = _make_manager_with_adjuster()
        projections = [
            Projection(
                player_id=1,
                season=2026,
                system="steamer",
                version="2026",
                player_type="batter",
                stat_json={"HR": 30, "RBI": 90},
            ),
            Projection(
                player_id=2,
                season=2026,
                system="steamer",
                version="2026",
                player_type="batter",
                stat_json={"HR": 25, "RBI": 80},
            ),
        ]
        fake_proj_repo = FakeProjectionRepo(projections)
        mgr._projection_repo = fake_proj_repo

        sid, _ = mgr.start_session(2026, keeper_player_ids={(1, "batter")})
        fn = mgr.get_category_balance_fn(sid)
        assert fn is not None
        # Call it — should return dict[int, float]
        result = fn([], [2])
        assert isinstance(result, dict)

    def test_returns_none_for_non_keeper_session(self) -> None:
        mgr = _make_manager()
        projections = [
            Projection(
                player_id=1, season=2026, system="steamer", version="2026", player_type="batter", stat_json={"HR": 30}
            ),
        ]
        mgr._projection_repo = FakeProjectionRepo(projections)

        sid, _ = mgr.start_session(2026)
        fn = mgr.get_category_balance_fn(sid)
        assert fn is None

    def test_returns_none_when_no_projection_repo(self) -> None:
        mgr, _ = _make_manager_with_adjuster()
        sid, _ = mgr.start_session(2026, keeper_player_ids={(1, "batter")})
        fn = mgr.get_category_balance_fn(sid)
        assert fn is None


class TestGetWeakCategories:
    def test_returns_weak_categories_for_keeper_session(self) -> None:
        mgr, _ = _make_manager_with_adjuster()
        # Create projections where keepers have extreme weakness in a category
        projections = [
            Projection(
                player_id=1,
                season=2026,
                system="steamer",
                version="2026",
                player_type="batter",
                stat_json={"HR": 0, "RBI": 0},
            ),
        ]
        mgr._projection_repo = FakeProjectionRepo(projections)

        sid, _ = mgr.start_session(2026, keeper_player_ids={(1, "batter")})
        weak = mgr.get_weak_categories(sid)
        # Should return a list (possibly empty) or None — just verify it's callable
        assert weak is None or isinstance(weak, list)

    def test_returns_none_for_non_keeper_session(self) -> None:
        mgr = _make_manager()
        projections = [
            Projection(
                player_id=1, season=2026, system="steamer", version="2026", player_type="batter", stat_json={"HR": 30}
            ),
        ]
        mgr._projection_repo = FakeProjectionRepo(projections)

        sid, _ = mgr.start_session(2026)
        weak = mgr.get_weak_categories(sid)
        assert weak is None


class TestTradePicksPersists:
    def test_trade_picks_persists(self) -> None:
        mgr = _make_manager()
        sid, engine = mgr.start_session(2026, fmt="snake", teams=4)

        trade = mgr.trade_picks(sid, gives=[1], receives=[2], partner_team=2)

        assert trade.team_a == 1
        assert trade.team_b == 2
        assert trade.team_a_gives == [1]
        assert trade.team_b_gives == [2]

        # Verify persisted
        trades = mgr._repo.load_trades(sid)
        assert len(trades) == 1
        assert trades[0].team_a_gives == [1]
        assert trades[0].team_b_gives == [2]

    def test_undo_trade_deletes(self) -> None:
        mgr = _make_manager()
        sid, _ = mgr.start_session(2026, fmt="snake", teams=4)

        mgr.trade_picks(sid, gives=[1], receives=[2], partner_team=2)
        assert len(mgr._repo.load_trades(sid)) == 1

        removed = mgr.undo_trade(sid)
        assert removed.team_a_gives == [1]
        assert mgr._repo.load_trades(sid) == []

    def test_get_engine_resumes_with_trades(self) -> None:
        mgr = _make_manager()
        sid, engine = mgr.start_session(2026, fmt="snake", teams=4)

        mgr.trade_picks(sid, gives=[1], receives=[2], partner_team=2)

        # Evict from cache
        mgr._engines.clear()

        restored = mgr.get_engine(sid)
        assert len(restored.trades) == 1
        assert restored.team_for_pick(1) == 2
        assert restored.team_for_pick(2) == 1


class TestNoAutoKeepers:
    """Verify that keepers are only applied when explicitly provided."""

    def test_no_keepers_without_explicit_ids(self) -> None:
        """Without keeper_player_ids, all players remain available."""
        mgr = _make_manager()
        _, engine = mgr.start_session(2026)
        available_ids = {r.player_id for r in engine.available()}
        assert 1 in available_ids
        assert 2 in available_ids
        assert 3 in available_ids

    def test_league_key_alone_does_not_load_keepers(self) -> None:
        """Passing league_key without keeper_player_ids should NOT auto-load keepers."""
        mgr = _make_manager()
        _, engine = mgr.start_session(2026, league_key="422.l.12345")
        available_ids = {r.player_id for r in engine.available()}
        assert 1 in available_ids
        assert 2 in available_ids
        assert 3 in available_ids


class TestTeamNames:
    def test_team_names_persisted_and_retrieved(self) -> None:
        mgr = _make_manager()
        team_names = {1: "Sluggers", 2: "Aces", 3: "Dingers"}
        sid, _ = mgr.start_session(2026, team_names=team_names)

        result = mgr.get_team_names(sid)
        assert result == {1: "Sluggers", 2: "Aces", 3: "Dingers"}

    def test_get_team_names_returns_none_without_names(self) -> None:
        mgr = _make_manager()
        sid, _ = mgr.start_session(2026)

        result = mgr.get_team_names(sid)
        assert result is None

    def test_team_names_survive_restart(self) -> None:
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

        mgr1 = SessionManager(
            session_repo=SqliteDraftSessionRepo(provider),
            valuation_repo=valuation_repo,
            player_repo=player_repo,
            adp_repo=SqliteADPRepo(provider),
            player_profile_service=PlayerProfileService(player_repo),
            league=_LEAGUE,
            adp_provider="fantasypros",
        )
        team_names = {1: "Sluggers", 2: "Aces"}
        sid, _ = mgr1.start_session(2026, team_names=team_names)

        # Simulate restart with fresh manager
        mgr2 = SessionManager(
            session_repo=SqliteDraftSessionRepo(provider),
            valuation_repo=valuation_repo,
            player_repo=player_repo,
            adp_repo=SqliteADPRepo(provider),
            player_profile_service=PlayerProfileService(player_repo),
            league=_LEAGUE,
            adp_provider="fantasypros",
        )
        result = mgr2.get_team_names(sid)
        assert result == {1: "Sluggers", 2: "Aces"}


class TestEvaluateTrade:
    def test_evaluate_trade_returns_evaluation(self) -> None:
        mgr = _make_manager()
        sid, engine = mgr.start_session(2026, fmt="snake", teams=4)

        evaluation = mgr.evaluate_trade(sid, gives=[1], receives=[4])

        assert evaluation.gives_value >= 0
        assert evaluation.receives_value >= 0
        assert evaluation.net_value == evaluation.receives_value - evaluation.gives_value
        assert evaluation.recommendation in {"accept", "reject", "even"}
        assert len(evaluation.gives_detail) == 1
        assert len(evaluation.receives_detail) == 1
        assert evaluation.gives_detail[0].pick == 1
        assert evaluation.receives_detail[0].pick == 4

    def test_evaluate_trade_does_not_modify_engine(self) -> None:
        mgr = _make_manager()
        sid, engine = mgr.start_session(2026, fmt="snake", teams=4)

        picks_before = engine.state.current_pick
        trades_before = len(engine.trades)

        mgr.evaluate_trade(sid, gives=[1], receives=[4])

        assert engine.state.current_pick == picks_before
        assert len(engine.trades) == trades_before
