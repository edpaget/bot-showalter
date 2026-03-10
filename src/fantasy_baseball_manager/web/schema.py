from collections.abc import AsyncGenerator  # noqa: TC003 — Strawberry resolves annotations at runtime
from typing import TYPE_CHECKING

import strawberry
from strawberry.types import Info  # noqa: TC002 — Strawberry needs this at runtime

from fantasy_baseball_manager.domain import ArbitrageReport, DraftBoard, Position, position_from_raw
from fantasy_baseball_manager.services import (
    DraftFormat,
    analyze_roster,
    auto_detect_position,
    build_arbitrage_report,
    build_draft_board,
    compute_scarcity,
    detect_falling_players,
    generate_tiers,
    recommend,
)
from fantasy_baseball_manager.web.types import (
    ADPReportType,
    ArbitrageAlertEvent,
    ArbitrageReportType,
    CategoryBalanceType,
    DraftBoardRowType,
    DraftBoardType,
    DraftEventType,
    DraftPickType,
    DraftSessionSummaryType,
    DraftStateType,
    FallingPlayerType,
    LeagueSettingsType,
    PickEvent,
    PickResultType,
    PlayerSummaryType,
    PlayerTierType,
    PositionScarcityType,
    ProjectionType,
    RecommendationType,
    RosterSlotType,
    SessionEvent,
    UndoEvent,
    ValuationType,
    WebConfigType,
    YahooPollStatusType,
)

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


def _build_pick_result(
    info: Info,
    session_id: int,
    pick: DraftPickType,
) -> PickResultType:
    mgr = _get_session_manager(info)
    engine = mgr.get_engine(session_id)

    recs = recommend(engine.state, limit=10)
    roster = engine.my_roster()
    needs = engine.my_needs()

    adp_lookup = {r.player_id: r.adp_overall for r in engine.state.available_pool.values() if r.adp_overall is not None}
    available = engine.available()
    report = build_arbitrage_report(
        engine.state.current_pick,
        available,
        engine.state.picks,
        adp_lookup,
    )

    return PickResultType(
        pick=pick,
        state=DraftStateType.from_state(session_id, engine.state),
        recommendations=[RecommendationType.from_domain(r) for r in recs],
        roster=[DraftPickType.from_domain(p) for p in roster],
        needs=[RosterSlotType(position=position_from_raw(pos), remaining=count) for pos, count in needs.items()],
        arbitrage=ArbitrageReportType.from_domain(report),
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

        board = build_draft_board(
            valuations,
            ctx.league,
            player_names,
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
        return WebConfigType.from_domain(ctx.web_config)

    @strawberry.field
    def session(self, info: Info, session_id: int) -> DraftStateType:
        mgr = _get_session_manager(info)
        engine = mgr.get_engine(session_id)
        return DraftStateType.from_state(session_id, engine.state)

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
        recs = recommend(engine.state, limit=limit)
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
        report = build_arbitrage_report(
            engine.state.current_pick,
            available,
            engine.state.picks,
            adp_lookup,
            threshold=threshold,
            limit=limit,
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
        )
        await ctx.event_bus.publish(
            session_id,
            SessionEvent(session_id=session_id, event_type="started"),
        )
        return DraftStateType.from_state(session_id, engine.state)

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
        falling = detect_falling_players(engine.state.current_pick, available, threshold=20, limit=20)
        alert_falling = [f for f in falling if f.value_rank <= 50][:3]
        if alert_falling:
            await ctx.event_bus.publish(
                session_id,
                ArbitrageAlertEvent(
                    session_id=session_id,
                    falling=[FallingPlayerType.from_domain(f) for f in alert_falling],
                ),
            )

        return _build_pick_result(info, session_id, pick_type)

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
