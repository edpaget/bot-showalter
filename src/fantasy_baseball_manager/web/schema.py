from collections.abc import AsyncGenerator  # noqa: TC003 — Strawberry resolves annotations at runtime
from typing import TYPE_CHECKING

import strawberry
from strawberry.types import Info  # noqa: TC002 — Strawberry needs this at runtime

from fantasy_baseball_manager.domain import (
    ArbitrageReport,
    DraftBoard,
    Position,
    TierAssignment,
    YahooDraftSetupInfo,
    position_from_raw,
)
from fantasy_baseball_manager.services import (
    DraftEngine,
    DraftFormat,
    KeeperPlannerService,
    PlayerEligibilityService,
    analyze_roster,
    auto_detect_position,
    build_arbitrage_report,
    build_draft_board,
    build_league_keeper_overview,
    compute_scarcity,
    derive_and_store_keeper_costs,
    derive_best_n_keeper_costs,
    detect_falling_players,
    ensure_prior_season_teams,
    generate_tiers,
    identify_needs,
    recommend,
)
from fantasy_baseball_manager.web.types import (
    ADPReportType,
    ArbitrageAlertEvent,
    ArbitrageReportType,
    CategoryBalanceType,
    CategoryNeedType,
    DraftBoardRowType,
    DraftBoardType,
    DraftEventType,
    DraftPickType,
    DraftSessionSummaryType,
    DraftStateType,
    DraftTradeType,
    FallingPlayerType,
    KeeperInfoType,
    KeeperPlanType,
    LeagueKeeperOverviewType,
    LeagueSettingsType,
    PickEvent,
    PickResultType,
    PickTradeEvaluationType,
    PlayerSummaryType,
    PlayerTierType,
    PositionScarcityType,
    ProjectionType,
    RecommendationType,
    RosterSlotType,
    SessionEvent,
    TradeEvent,
    UndoEvent,
    ValuationType,
    WebConfigType,
    YahooDraftSetupInfoType,
    YahooPollStatusType,
    YahooRosterType,
    YahooStandingsEntryType,
    YahooTeamType,
)
from fantasy_baseball_manager.yahoo.draft_source import YahooDraftSource
from fantasy_baseball_manager.yahoo.league_source import YahooLeagueSource
from fantasy_baseball_manager.yahoo.roster_source import YahooRosterSource

if TYPE_CHECKING:
    from fantasy_baseball_manager.web.app import AppContext
    from fantasy_baseball_manager.web.session_manager import SessionManager


def _get_context(info: Info) -> AppContext:
    return info.context["app_context"]


def _get_session_manager(info: Info) -> SessionManager:
    ctx = _get_context(info)
    mgr = ctx.session_manager
    if mgr is None:
        msg = "Session management is not configured"
        raise ValueError(msg)
    return mgr


def _get_keeper_count(mgr: SessionManager, session_id: int) -> int:
    return len(mgr.get_keepers(session_id))


def _compute_arbitrage_cat_scores(
    info: Info,
    session_id: int,
    engine: DraftEngine,
) -> dict[int, float] | None:
    """Compute category balance scores for arbitrage boosting in keeper sessions."""
    mgr = _get_session_manager(info)
    cat_bal_fn = mgr.get_category_balance_fn(session_id)
    if cat_bal_fn is None:
        return None
    roster_ids = [p.player_id for p in engine.my_roster()]
    available_ids = list(engine.state.available_pool.keys())
    return cat_bal_fn(roster_ids, available_ids)


def _build_keeper_planner(ctx: AppContext, season: int, league_name: str) -> KeeperPlannerService:
    """Build a fresh KeeperPlannerService from current DB state."""
    system = ctx.default_system
    version = ctx.default_version
    keeper_costs = ctx.container.keeper_cost_repo.find_by_season_league(season, league_name)
    valuations = ctx.container.valuation_repo.get_by_season(season, system=system, version=version)
    players = ctx.container.player_repo.get_by_ids([v.player_id for v in valuations])
    projections = ctx.container.projection_repo.get_by_season(season)
    eligibility = PlayerEligibilityService(
        ctx.container.position_appearance_repo,
        pitching_stats_repo=ctx.container.pitching_stats_repo,
    )
    batter_positions = eligibility.get_batter_positions(season, ctx.league)
    pitcher_projs = [p for p in projections if p.player_type == "pitcher"]
    pitcher_ids = [p.player_id for p in pitcher_projs]
    pitcher_positions = eligibility.get_pitcher_positions(season, ctx.league, pitcher_ids, projections=pitcher_projs)
    return KeeperPlannerService(
        keeper_costs=keeper_costs,
        valuations=valuations,
        players=players,
        projections=projections,
        league=ctx.league,
        batter_positions=batter_positions,
        pitcher_positions=pitcher_positions,
    )


def _resolve_prior_league_key(ctx: AppContext, league_key: str, season: int) -> tuple[str, str]:
    """Resolve the prior-season league key and game key.

    Returns (prior_league_key, prior_game_key).
    """
    yahoo = ctx.yahoo_web_context
    assert yahoo is not None  # noqa: S101
    prior_season = season - 1
    stored_league = yahoo.league_repo.get_by_league_key(league_key)
    if stored_league is not None and stored_league.renew is not None:
        prior_league_key = stored_league.renew.replace("_", ".l.", 1)
        prior_game_key = prior_league_key.split(".l.")[0]
        return prior_league_key, prior_game_key
    prior_game_key = yahoo.client.get_game_key(prior_season)
    league_id = league_key.split(".l.")[1]
    return f"{prior_game_key}.l.{league_id}", prior_game_key


def _derive_keeper_costs_from_yahoo(
    ctx: AppContext, season: int, league_key: str | None = None, cost_floor: float = 1.0
) -> None:
    """Auto-derive keeper costs from Yahoo roster data and commit."""
    yahoo = ctx.yahoo_web_context
    if yahoo is None:
        return
    team_repo = ctx.yahoo_team_repo
    if team_repo is None:
        return

    if league_key is None:
        league_key = ctx.yahoo_league_info.league_key if ctx.yahoo_league_info else None
    if league_key is None:
        return

    prior_league_key, prior_game_key = _resolve_prior_league_key(ctx, league_key, season)
    prior_season = season - 1

    # Sync prior-season teams if not already in DB
    league_source = YahooLeagueSource(yahoo.client)
    ensure_prior_season_teams(
        team_repo=team_repo,
        league_source=league_source,
        league_repo=yahoo.league_repo,
        prior_league_key=prior_league_key,
        prior_game_key=prior_game_key,
    )
    with yahoo.provider.connection() as conn:
        conn.commit()

    roster_source = YahooRosterSource(yahoo.client, yahoo.player_mapper, roster_repo=ctx.yahoo_roster_repo)
    if yahoo.league_config.keeper_format == "best_n":
        derive_best_n_keeper_costs(
            roster_source=roster_source,
            team_repo=team_repo,
            keeper_repo=ctx.container.keeper_cost_repo,
            prior_league_key=prior_league_key,
            prior_season=prior_season,
            season=season,
            league_name=yahoo.league_name,
        )
    else:
        draft_source = YahooDraftSource(yahoo.client, yahoo.player_mapper)
        derive_and_store_keeper_costs(
            draft_source=draft_source,
            roster_source=roster_source,
            team_repo=team_repo,
            keeper_repo=ctx.container.keeper_cost_repo,
            league_key=league_key,
            prior_league_key=prior_league_key,
            prior_season=prior_season,
            season=season,
            league_name=yahoo.league_name,
            cost_floor=cost_floor,
        )

    with yahoo.provider.connection() as conn:
        conn.commit()


def _ensure_keeper_planner(ctx: AppContext, season: int) -> KeeperPlannerService | None:
    """Return the keeper planner, auto-deriving costs from Yahoo if needed."""
    planner = ctx.keeper_planner_ref.planner
    if planner is not None:
        return planner

    # No planner yet — try to auto-derive from Yahoo if configured
    yahoo = ctx.yahoo_web_context
    if yahoo is None:
        return None

    # Only derive if no costs exist yet — avoids redundant Yahoo API calls
    existing = ctx.container.keeper_cost_repo.find_by_season_league(season, yahoo.league_name)
    if not existing:
        _derive_keeper_costs_from_yahoo(ctx, season)

    planner = _build_keeper_planner(ctx, season, yahoo.league_name)
    ctx.keeper_planner_ref.planner = planner
    return planner


def _build_pick_result(
    info: Info,
    session_id: int,
    pick: DraftPickType,
    *,
    arb_cat_scores: dict[int, float] | None = None,
) -> PickResultType:
    mgr = _get_session_manager(info)
    engine = mgr.get_engine(session_id)

    # Compute category scores once if not pre-computed (e.g. from undo)
    if arb_cat_scores is None:
        arb_cat_scores = _compute_arbitrage_cat_scores(info, session_id, engine)

    weak_cats = mgr.get_weak_categories(session_id)
    recs = recommend(engine.state, limit=10, cat_scores=arb_cat_scores or {}, weak_categories=weak_cats)
    roster = engine.my_roster()
    needs = engine.my_needs()

    adp_lookup = {r.player_id: r.adp_overall for r in engine.state.available_pool.values() if r.adp_overall is not None}
    available = engine.available()
    report = build_arbitrage_report(
        engine.state.current_pick,
        available,
        engine.state.picks,
        adp_lookup,
        category_scores=arb_cat_scores,
    )

    # Compute category balance and needs for the updated roster
    ctx = _get_context(info)
    roster_ids = [p.player_id for p in roster]
    if roster_ids:
        projections = ctx.container.projection_repo.get_by_season(engine.state.config.season)
        analysis = analyze_roster(roster_ids, projections, ctx.league)
        balance_types = [CategoryBalanceType.from_domain(p) for p in analysis.projections]

        available_ids = [r.player_id for r in available]
        player_ids = {*roster_ids, *available_ids}
        players = ctx.container.player_repo.get_by_ids(list(player_ids))
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}
        cat_needs = identify_needs(
            roster_ids,
            available_ids,
            projections,
            ctx.league,
            player_names,
            top_n=5,
        )
        category_needs_types = [CategoryNeedType.from_domain(n) for n in cat_needs]
    else:
        balance_types = []
        category_needs_types = []

    keeper_count = _get_keeper_count(mgr, session_id)
    return PickResultType(
        pick=pick,
        state=DraftStateType.from_state(session_id, engine.state, keeper_count=keeper_count, trades=engine.trades),
        recommendations=[RecommendationType.from_domain(r) for r in recs],
        roster=[DraftPickType.from_domain(p) for p in roster],
        needs=[RosterSlotType(position=position_from_raw(pos), remaining=count) for pos, count in needs.items()],
        arbitrage=ArbitrageReportType.from_domain(report),
        balance=balance_types,
        category_needs=category_needs_types,
    )


@strawberry.type
class Query:
    @strawberry.field
    def board(
        self,
        info: Info,
        season: int,
        system: str | None = None,
        version: str | None = None,
        player_type: str | None = None,
        position: Position | None = None,
        top: int | None = None,
    ) -> DraftBoardType:
        ctx = _get_context(info)
        system = system or ctx.default_system
        version = version or ctx.default_version
        valuations = ctx.container.valuation_repo.get_by_season(season, system=system, version=version)

        if player_type is not None:
            valuations = [v for v in valuations if v.player_type == player_type]
        if position is not None:
            valuations = [v for v in valuations if v.position == position.value]

        player_ids = [v.player_id for v in valuations]
        players = ctx.container.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        adp_list = ctx.container.adp_repo.get_by_season(season, provider=ctx.adp_provider)
        profiles = ctx.container.player_profile_service.enrich_valuations(valuations, season)

        tier_results = generate_tiers(valuations, ctx.container.player_repo)
        tier_assignments = [TierAssignment(player_id=t.player_id, tier=t.tier) for t in tier_results]

        board = build_draft_board(
            valuations,
            ctx.league,
            player_names,
            tiers=tier_assignments,
            adp=adp_list if adp_list else None,
            profiles=profiles,
            breakout_predictions=ctx.breakout_predictions,
        )

        if top is not None:
            board = DraftBoard(
                rows=board.rows[:top],
                batting_categories=board.batting_categories,
                pitching_categories=board.pitching_categories,
            )

        return DraftBoardType.from_domain(board)

    @strawberry.field
    def tiers(
        self,
        info: Info,
        season: int,
        system: str | None = None,
        version: str | None = None,
        player_type: str | None = None,
        method: str = "gap",
        max_tiers: int = 5,
    ) -> list[PlayerTierType]:
        ctx = _get_context(info)
        system = system or ctx.default_system
        version = version or ctx.default_version
        valuations = ctx.container.valuation_repo.get_by_season(season, system=system, version=version)

        if player_type is not None:
            valuations = [v for v in valuations if v.player_type == player_type]

        result = generate_tiers(valuations, ctx.container.player_repo, method=method, max_tiers=max_tiers)
        return [PlayerTierType.from_domain(t) for t in result]

    @strawberry.field
    def scarcity(
        self,
        info: Info,
        season: int,
        system: str | None = None,
        version: str | None = None,
    ) -> list[PositionScarcityType]:
        ctx = _get_context(info)
        system = system or ctx.default_system
        version = version or ctx.default_version
        valuations = ctx.container.valuation_repo.get_by_season(season, system=system, version=version)

        result = compute_scarcity(valuations, ctx.league)
        return [PositionScarcityType.from_domain(s) for s in result]

    @strawberry.field
    def league(self, info: Info) -> LeagueSettingsType:
        ctx = _get_context(info)
        return LeagueSettingsType.from_domain(ctx.league)

    @strawberry.field
    def web_config(self, info: Info) -> WebConfigType:
        ctx = _get_context(info)
        return WebConfigType.from_domain(ctx.web_config, yahoo_league_info=ctx.yahoo_league_info)

    @strawberry.field
    def session(self, info: Info, session_id: int) -> DraftStateType:
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)
        keeper_count = _get_keeper_count(mgr, session_id)
        return DraftStateType.from_state(session_id, engine.state, keeper_count=keeper_count, trades=engine.trades)

    @strawberry.field
    def keepers(self, info: Info, session_id: int) -> list[KeeperInfoType]:
        mgr = _get_session_manager(info)
        snapshot = mgr.get_keepers(session_id)
        return [KeeperInfoType.from_dict(k) for k in snapshot]

    @strawberry.field
    def sessions(
        self,
        info: Info,
        league: str | None = None,
        season: int | None = None,
        status: str | None = None,
    ) -> list[DraftSessionSummaryType]:
        mgr = _get_session_manager(info)
        summaries = mgr.list_sessions(league=league, season=season, status=status)
        return [DraftSessionSummaryType.from_domain(s.record, s.pick_count) for s in summaries]

    @strawberry.field
    def recommendations(
        self,
        info: Info,
        session_id: int,
        position: Position | None = None,
        limit: int = 10,
    ) -> list[RecommendationType]:
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)
        cat_bal_fn = mgr.get_category_balance_fn(session_id)
        weak_cats = mgr.get_weak_categories(session_id)
        recs = recommend(engine.state, limit=limit, category_balance_fn=cat_bal_fn, weak_categories=weak_cats)
        if position is not None:
            recs = [r for r in recs if r.position == position.value]
        return [RecommendationType.from_domain(r) for r in recs]

    @strawberry.field
    def roster(
        self,
        info: Info,
        session_id: int,
        team: int | None = None,
    ) -> list[DraftPickType]:
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)
        t = team if team is not None else engine.state.config.user_team
        return [DraftPickType.from_domain(p) for p in engine.state.team_rosters[t]]

    @strawberry.field
    def needs(self, info: Info, session_id: int) -> list[RosterSlotType]:
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)
        return [
            RosterSlotType(position=position_from_raw(pos), remaining=count) for pos, count in engine.my_needs().items()
        ]

    @strawberry.field
    def balance(self, info: Info, session_id: int) -> list[CategoryBalanceType]:
        ctx = _get_context(info)
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)

        roster = engine.my_roster()
        if not roster:
            return []

        roster_ids = [p.player_id for p in roster]
        projections = ctx.container.projection_repo.get_by_season(engine.state.config.season)
        analysis = analyze_roster(roster_ids, projections, ctx.league)
        return [CategoryBalanceType.from_domain(p) for p in analysis.projections]

    @strawberry.field
    def category_needs(
        self,
        info: Info,
        session_id: int,
        top_n: int = 5,
    ) -> list[CategoryNeedType]:
        ctx = _get_context(info)
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)

        roster = engine.my_roster()
        if not roster:
            return []

        roster_ids = [p.player_id for p in roster]
        available_rows = engine.available()
        available_ids = [r.player_id for r in available_rows]
        projections = ctx.container.projection_repo.get_by_season(engine.state.config.season)

        player_ids = {*roster_ids, *available_ids}
        players = ctx.container.player_repo.get_by_ids(list(player_ids))
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        needs = identify_needs(
            roster_ids,
            available_ids,
            projections,
            ctx.league,
            player_names,
            top_n=top_n,
        )
        return [CategoryNeedType.from_domain(n) for n in needs]

    @strawberry.field
    def available(
        self,
        info: Info,
        session_id: int,
        position: Position | None = None,
        limit: int = 50,
    ) -> list[DraftBoardRowType]:
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)
        rows = engine.available(position.value if position is not None else None)[:limit]
        return [DraftBoardRowType.from_domain(r) for r in rows]

    @strawberry.field
    def arbitrage(
        self,
        info: Info,
        session_id: int,
        threshold: int = 10,
        position: Position | None = None,
        limit: int = 20,
    ) -> ArbitrageReportType:
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)
        adp_lookup = {
            r.player_id: r.adp_overall for r in engine.state.available_pool.values() if r.adp_overall is not None
        }
        available = engine.available()
        arb_cat_scores = _compute_arbitrage_cat_scores(info, session_id, engine)
        report = build_arbitrage_report(
            engine.state.current_pick,
            available,
            engine.state.picks,
            adp_lookup,
            threshold=threshold,
            limit=limit,
            category_scores=arb_cat_scores,
        )
        if position is not None:
            filtered_falling = [f for f in report.falling if f.position == position.value]
            report = ArbitrageReport(
                current_pick=report.current_pick,
                falling=filtered_falling,
                reaches=report.reaches,
            )
        return ArbitrageReportType.from_domain(report)

    @strawberry.field
    def projection_board(
        self,
        info: Info,
        season: int,
        system: str,
        version: str,
        player_type: str | None = None,
    ) -> list[ProjectionType]:
        ctx = _get_context(info)
        results = ctx.container.projection_lookup_service.browse(season, system, version, player_type=player_type)
        return [ProjectionType.from_domain(p) for p in results]

    @strawberry.field
    def projections(
        self,
        info: Info,
        season: int,
        player_name: str,
        system: str | None = None,
    ) -> list[ProjectionType]:
        ctx = _get_context(info)
        results = ctx.container.projection_lookup_service.lookup(player_name, season, system=system)
        allow = ctx.web_config.projections
        if allow:
            results = [r for r in results if any(sv.system == r.system and sv.version == r.version for sv in allow)]
        return [ProjectionType.from_domain(p) for p in results]

    @strawberry.field
    def valuations(
        self,
        info: Info,
        season: int,
        system: str | None = None,
        version: str | None = None,
        player_type: str | None = None,
        position: str | None = None,
        top: int | None = None,
    ) -> list[ValuationType]:
        ctx = _get_context(info)
        results = ctx.container.valuation_lookup_service.rankings(
            season,
            system=system,
            version=version,
            player_type=player_type,
            position=position,
            top=top,
        )
        allow = ctx.web_config.valuations
        if allow:
            results = [r for r in results if any(sv.system == r.system and sv.version == r.version for sv in allow)]
        return [ValuationType.from_domain(v) for v in results]

    @strawberry.field
    def adp_report(
        self,
        info: Info,
        season: int,
        system: str | None = None,
        version: str | None = None,
        provider: str | None = None,
    ) -> ADPReportType:
        ctx = _get_context(info)
        system = system or ctx.default_system
        version = version or ctx.default_version
        provider = provider or ctx.adp_provider
        report = ctx.container.adp_report_service.compute_value_over_adp(
            season,
            system,
            version,
            provider=provider,
        )
        return ADPReportType.from_domain(report)

    @strawberry.field
    def player_search(
        self,
        info: Info,
        name: str,
        season: int,
    ) -> list[PlayerSummaryType]:
        ctx = _get_context(info)
        results = ctx.container.player_bio_service.search(name, season)
        return [PlayerSummaryType.from_domain(s) for s in results]

    @strawberry.field
    def player_bio(
        self,
        info: Info,
        player_id: int,
        season: int,
    ) -> PlayerSummaryType | None:
        ctx = _get_context(info)
        result = ctx.container.player_bio_service.get_bio(player_id, season)
        if result is None:
            return None
        return PlayerSummaryType.from_domain(result)

    @strawberry.field
    def plan_keeper_draft(
        self,
        info: Info,
        season: int,
        max_keepers: int,
        system: str | None = None,
        version: str | None = None,
        custom_scenarios: list[list[int]] | None = None,
        board_preview_size: int = 20,
    ) -> KeeperPlanType:
        ctx = _get_context(info)
        keeper_planner = _ensure_keeper_planner(ctx, season)
        if keeper_planner is None:
            msg = "Keeper planner is not configured"
            raise ValueError(msg)
        converted = [set(s) for s in custom_scenarios] if custom_scenarios else None
        result = keeper_planner.plan(
            season=season,
            max_keepers=max_keepers,
            custom_scenarios=converted,
            board_preview_size=board_preview_size,
        )
        return KeeperPlanType.from_domain(result)

    @strawberry.field
    def yahoo_poll_status(self, info: Info, session_id: int) -> YahooPollStatusType:
        ctx = _get_context(info)
        ypm = ctx.yahoo_poller_manager
        if ypm is None:
            msg = "Yahoo polling is not configured"
            raise ValueError(msg)
        status = ypm.get_status(session_id)
        return YahooPollStatusType(
            active=status.active,
            last_poll_at=status.last_poll_at,
            picks_ingested=status.picks_ingested,
        )

    @strawberry.field
    def yahoo_teams(self, info: Info, league_key: str) -> list[YahooTeamType]:
        ctx = _get_context(info)
        repo = ctx.yahoo_team_repo
        if repo is None:
            msg = "Yahoo league is not configured"
            raise ValueError(msg)
        teams = repo.get_by_league_key(league_key)
        return [YahooTeamType.from_domain(t) for t in teams]

    @strawberry.field
    def yahoo_standings(self, info: Info, league_key: str, season: int) -> list[YahooStandingsEntryType]:
        ctx = _get_context(info)
        repo = ctx.yahoo_team_stats_repo
        if repo is None:
            msg = "Yahoo league is not configured"
            raise ValueError(msg)
        stats = repo.get_by_league_season(league_key, season)
        return [YahooStandingsEntryType.from_domain(s) for s in stats]

    @strawberry.field
    def yahoo_rosters(self, info: Info, league_key: str) -> list[YahooRosterType]:
        ctx = _get_context(info)
        repo = ctx.yahoo_roster_repo
        if repo is None:
            msg = "Yahoo league is not configured"
            raise ValueError(msg)
        rosters = repo.get_by_league_latest(league_key)
        return [YahooRosterType.from_domain(r) for r in rosters]

    @strawberry.field
    def yahoo_roster(self, info: Info, team_key: str, league_key: str) -> YahooRosterType | None:
        ctx = _get_context(info)
        repo = ctx.yahoo_roster_repo
        if repo is None:
            msg = "Yahoo league is not configured"
            raise ValueError(msg)
        roster = repo.get_latest_by_team(team_key, league_key)
        if roster is None:
            return None
        return YahooRosterType.from_domain(roster)

    @strawberry.field
    def yahoo_draft_setup(self, info: Info, league_key: str, season: int) -> YahooDraftSetupInfoType:
        ctx = _get_context(info)
        league_repo = ctx.yahoo_league_repo
        team_repo = ctx.yahoo_team_repo
        if league_repo is None or team_repo is None:
            msg = "Yahoo league is not configured"
            raise ValueError(msg)

        league = league_repo.get_by_league_key(league_key)
        if league is None:
            msg = f"League {league_key} has not been synced"
            raise ValueError(msg)

        teams = team_repo.get_by_league_key(league_key)
        if not teams:
            msg = f"No teams found for league {league_key}"
            raise ValueError(msg)

        user_team = next((t for t in teams if t.is_owned_by_user), None)
        if user_team is None:
            msg = f"No user team found for league {league_key}"
            raise ValueError(msg)

        team_names = {t.team_id: t.name for t in teams}

        # Determine draft format from stored draft_type
        draft_type_lower = league.draft_type.lower()
        if "auction" in draft_type_lower:
            draft_format = "auction"
        elif "snake" in draft_type_lower:
            draft_format = "snake"
        else:
            draft_format = "live"

        # Look up keeper data
        keeper_player_ids: list[int] = []
        max_keepers: int | None = None
        if league.is_keeper:
            yahoo_ctx = ctx.yahoo_web_context
            if yahoo_ctx is not None and yahoo_ctx.league_config.max_keepers is not None:
                max_keepers = yahoo_ctx.league_config.max_keepers
            costs = ctx.container.keeper_cost_repo.find_by_season_league(season, league.name)
            keeper_player_ids = [c.player_id for c in costs]

        setup = YahooDraftSetupInfo(
            num_teams=league.num_teams,
            draft_format=draft_format,
            user_team_id=user_team.team_id,
            team_names=team_names,
            draft_order=[],
            is_keeper=league.is_keeper,
            max_keepers=max_keepers,
            keeper_player_ids=keeper_player_ids,
        )
        return YahooDraftSetupInfoType.from_domain(setup)

    @strawberry.field
    def yahoo_keeper_overview(
        self,
        info: Info,
        league_key: str,
        season: int,
        max_keepers: int,
    ) -> LeagueKeeperOverviewType:
        ctx = _get_context(info)
        repo = ctx.yahoo_roster_repo
        if repo is None:
            msg = "Yahoo league is not configured"
            raise ValueError(msg)

        # Rosters and teams are stored under the prior-season league key
        prior_league_key, _prior_game_key = _resolve_prior_league_key(ctx, league_key, season)

        rosters = repo.get_by_league_latest(prior_league_key)
        valuations = ctx.container.valuation_repo.get_by_season(
            season,
            system=ctx.default_system,
            version=ctx.default_version,
        )
        players = ctx.container.player_repo.all()

        team_repo = ctx.yahoo_team_repo
        teams = team_repo.get_by_league_key(prior_league_key) if team_repo else []
        team_names = {t.team_key: t.name for t in teams}
        user_team = next((t for t in teams if t.is_owned_by_user), None)
        user_team_key = user_team.team_key if user_team else ""

        overview = build_league_keeper_overview(
            rosters=rosters,
            valuations=valuations,
            players=players,
            max_keepers=max_keepers,
            user_team_key=user_team_key,
            team_names=team_names,
        )
        return LeagueKeeperOverviewType.from_domain(overview)

    @strawberry.field
    def evaluate_trade(
        self,
        info: Info,
        session_id: int,
        gives: list[int],
        receives: list[int],
    ) -> PickTradeEvaluationType:
        mgr = _get_session_manager(info)
        evaluation = mgr.evaluate_trade(session_id, gives, receives)
        return PickTradeEvaluationType.from_domain(evaluation)


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def start_session(
        self,
        info: Info,
        season: int,
        system: str | None = None,
        version: str | None = None,
        teams: int | None = None,
        user_team: int = 1,
        format: str = "snake",
        budget: int | None = None,
        keeper_player_ids: list[int] | None = None,
    ) -> DraftStateType:
        ctx = _get_context(info)
        system = system or ctx.default_system
        version = version or ctx.default_version
        mgr = _get_session_manager(info)
        session_id, engine = mgr.start_session(
            season,
            system=system,
            version=version,
            teams=teams,
            user_team=user_team,
            fmt=format,
            budget=budget,
            keeper_player_ids=set(keeper_player_ids) if keeper_player_ids else None,
        )
        await ctx.event_bus.publish(
            session_id,
            SessionEvent(session_id=session_id, event_type="started"),
        )
        keeper_count = _get_keeper_count(mgr, session_id)
        return DraftStateType.from_state(session_id, engine.state, keeper_count=keeper_count, trades=engine.trades)

    @strawberry.mutation
    async def pick(
        self,
        info: Info,
        session_id: int,
        player_id: int,
        position: Position,
        price: int | None = None,
        team: int | None = None,
    ) -> PickResultType:
        ctx = _get_context(info)
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)

        pos_str = position.value

        if pos_str not in engine.state.config.roster_slots:
            # Try auto-detecting from the player's data (e.g., "P" → "SP"/"RP")
            player = engine.state.available_pool.get(player_id)
            if player is not None:
                detected = auto_detect_position(player, engine.my_needs(), engine.state.config.roster_slots)
                if detected is not None:
                    pos_str = detected

        if team is None:
            if engine.state.config.format == DraftFormat.SNAKE:
                team = engine.team_on_clock()
            else:
                team = engine.state.config.user_team

        draft_pick = mgr.pick(session_id, player_id, team, pos_str, price=price)
        pick_type = DraftPickType.from_domain(draft_pick)
        await ctx.event_bus.publish(
            session_id,
            PickEvent(pick=pick_type, session_id=session_id),
        )

        # Publish arbitrage alert for significant fallers
        available = engine.available()
        arb_cat_scores = _compute_arbitrage_cat_scores(info, session_id, engine)
        falling = detect_falling_players(
            engine.state.current_pick, available, threshold=20, limit=20, category_scores=arb_cat_scores
        )
        alert_falling = [f for f in falling if f.value_rank <= 50][:3]
        if alert_falling:
            await ctx.event_bus.publish(
                session_id,
                ArbitrageAlertEvent(
                    session_id=session_id,
                    falling=[FallingPlayerType.from_domain(f) for f in alert_falling],
                ),
            )

        return _build_pick_result(info, session_id, pick_type, arb_cat_scores=arb_cat_scores)

    @strawberry.mutation
    async def undo(self, info: Info, session_id: int) -> PickResultType:
        ctx = _get_context(info)
        mgr = _get_session_manager(info)
        undone = mgr.undo(session_id)
        pick_type = DraftPickType.from_domain(undone)
        await ctx.event_bus.publish(
            session_id,
            UndoEvent(pick=pick_type, session_id=session_id),
        )
        return _build_pick_result(info, session_id, pick_type)

    @strawberry.mutation
    async def end_session(self, info: Info, session_id: int) -> bool:
        ctx = _get_context(info)
        mgr = _get_session_manager(info)
        # Stop Yahoo polling if active
        if ctx.yahoo_poller_manager is not None:
            await ctx.yahoo_poller_manager.stop_polling(session_id)
        mgr.end_session(session_id)
        await ctx.event_bus.publish(
            session_id,
            SessionEvent(session_id=session_id, event_type="ended"),
        )
        return True

    @strawberry.mutation
    async def trade_picks(
        self,
        info: Info,
        session_id: int,
        gives: list[int],
        receives: list[int],
        partner_team: int,
        team_a: int | None = None,
    ) -> DraftStateType:
        ctx = _get_context(info)
        mgr = _get_session_manager(info)
        trade = mgr.trade_picks(session_id, gives, receives, partner_team, team_a=team_a)
        engine = mgr.get_engine(session_id)
        keeper_count = _get_keeper_count(mgr, session_id)
        state = DraftStateType.from_state(session_id, engine.state, keeper_count=keeper_count, trades=engine.trades)
        await ctx.event_bus.publish(
            session_id,
            TradeEvent(
                session_id=session_id,
                trade=DraftTradeType.from_domain(trade),
                action="trade",
                state=state,
            ),
        )
        return state

    @strawberry.mutation
    async def undo_trade(self, info: Info, session_id: int) -> DraftStateType:
        ctx = _get_context(info)
        mgr = _get_session_manager(info)
        removed = mgr.undo_trade(session_id)
        engine = mgr.get_engine(session_id)
        keeper_count = _get_keeper_count(mgr, session_id)
        state = DraftStateType.from_state(session_id, engine.state, keeper_count=keeper_count, trades=engine.trades)
        await ctx.event_bus.publish(
            session_id,
            TradeEvent(
                session_id=session_id,
                trade=DraftTradeType.from_domain(removed),
                action="undo",
                state=state,
            ),
        )
        return state

    @strawberry.mutation
    async def start_yahoo_poll(
        self,
        info: Info,
        session_id: int,
        league_key: str,
    ) -> bool:
        ctx = _get_context(info)
        ypm = ctx.yahoo_poller_manager
        if ypm is None:
            msg = "Yahoo polling is not configured"
            raise ValueError(msg)
        return await ypm.start_polling(session_id, league_key)

    @strawberry.mutation
    async def stop_yahoo_poll(self, info: Info, session_id: int) -> bool:
        ctx = _get_context(info)
        ypm = ctx.yahoo_poller_manager
        if ypm is None:
            msg = "Yahoo polling is not configured"
            raise ValueError(msg)
        return await ypm.stop_polling(session_id)

    @strawberry.mutation
    async def derive_keeper_costs(
        self,
        info: Info,
        league_key: str,
        season: int,
        cost_floor: float | None = None,
    ) -> int:
        ctx = _get_context(info)
        yahoo = ctx.yahoo_web_context
        if yahoo is None:
            msg = "Yahoo league is not configured"
            raise ValueError(msg)

        _derive_keeper_costs_from_yahoo(
            ctx, season, league_key=league_key, cost_floor=cost_floor if cost_floor is not None else 1.0
        )

        # Rebuild keeper planner with fresh data
        ctx.keeper_planner_ref.planner = _build_keeper_planner(ctx, season, yahoo.league_name)

        # Return count of derived costs
        costs = ctx.container.keeper_cost_repo.find_by_season_league(season, yahoo.league_name)
        return len(costs)


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def draft_events(
        self,
        info: Info,
        session_id: int,
    ) -> AsyncGenerator[DraftEventType]:
        ctx = _get_context(info)
        q = ctx.event_bus.subscribe(session_id)
        try:
            while True:
                event = await q.get()
                yield event
        finally:
            ctx.event_bus.unsubscribe(session_id, q)
