from typing import TYPE_CHECKING

import strawberry
from strawberry.types import Info  # noqa: TC002 — Strawberry needs this at runtime

from fantasy_baseball_manager.domain import DraftBoard
from fantasy_baseball_manager.services import (
    DraftFormat,
    analyze_roster,
    build_draft_board,
    compute_scarcity,
    generate_tiers,
    recommend,
)
from fantasy_baseball_manager.web.types import (
    CategoryBalanceType,
    DraftBoardRowType,
    DraftBoardType,
    DraftPickType,
    DraftSessionSummaryType,
    DraftStateType,
    LeagueSettingsType,
    PickResultType,
    PlayerTierType,
    PositionScarcityType,
    RecommendationType,
    RosterSlotType,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.web.app import AppContext


def _get_context(info: Info) -> AppContext:
    return info.context["app_context"]


def _build_pick_result(
    info: Info,
    session_id: int,
    pick: DraftPickType,
) -> PickResultType:
    ctx = _get_context(info)
    mgr = ctx.session_manager
    assert mgr is not None  # noqa: S101
    engine = mgr.get_engine(session_id)

    recs = recommend(engine.state, limit=10)
    roster = engine.my_roster()
    needs = engine.my_needs()

    return PickResultType(
        pick=pick,
        state=DraftStateType.from_state(session_id, engine.state),
        recommendations=[RecommendationType.from_domain(r) for r in recs],
        roster=[DraftPickType.from_domain(p) for p in roster],
        needs=[RosterSlotType(position=pos, remaining=count) for pos, count in needs.items()],
    )


@strawberry.type
class Query:
    @strawberry.field
    def board(
        self,
        info: Info,
        season: int,
        system: str = "zar",
        version: str = "1.0",
        player_type: str | None = None,
        position: str | None = None,
        top: int | None = None,
    ) -> DraftBoardType:
        ctx = _get_context(info)
        valuations = ctx.container.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        if player_type is not None:
            valuations = [v for v in valuations if v.player_type == player_type]
        if position is not None:
            valuations = [v for v in valuations if v.position == position]

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
        system: str = "zar",
        version: str = "1.0",
        player_type: str | None = None,
        method: str = "gap",
        max_tiers: int = 5,
    ) -> list[PlayerTierType]:
        ctx = _get_context(info)
        valuations = ctx.container.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        if player_type is not None:
            valuations = [v for v in valuations if v.player_type == player_type]

        result = generate_tiers(valuations, ctx.container.player_repo, method=method, max_tiers=max_tiers)
        return [PlayerTierType.from_domain(t) for t in result]

    @strawberry.field
    def scarcity(
        self,
        info: Info,
        season: int,
        system: str = "zar",
        version: str = "1.0",
    ) -> list[PositionScarcityType]:
        ctx = _get_context(info)
        valuations = ctx.container.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        result = compute_scarcity(valuations, ctx.league)
        return [PositionScarcityType.from_domain(s) for s in result]

    @strawberry.field
    def league(self, info: Info) -> LeagueSettingsType:
        ctx = _get_context(info)
        return LeagueSettingsType.from_domain(ctx.league)

    @strawberry.field
    def session(self, info: Info, session_id: int) -> DraftStateType:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
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
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        records = mgr._repo.list_sessions(league=league, season=season)
        if status is not None:
            records = [r for r in records if r.status == status]
        return [DraftSessionSummaryType.from_domain(r, mgr._repo.count_picks(r.id or 0)) for r in records]

    @strawberry.field
    def recommendations(
        self,
        info: Info,
        session_id: int,
        position: str | None = None,
        limit: int = 10,
    ) -> list[RecommendationType]:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        engine = mgr.get_engine(session_id)
        recs = recommend(engine.state, limit=limit)
        if position is not None:
            recs = [r for r in recs if r.position == position]
        return [RecommendationType.from_domain(r) for r in recs]

    @strawberry.field
    def roster(
        self,
        info: Info,
        session_id: int,
        team: int | None = None,
    ) -> list[DraftPickType]:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        engine = mgr.get_engine(session_id)
        t = team if team is not None else engine.state.config.user_team
        return [DraftPickType.from_domain(p) for p in engine.state.team_rosters[t]]

    @strawberry.field
    def needs(self, info: Info, session_id: int) -> list[RosterSlotType]:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        engine = mgr.get_engine(session_id)
        return [RosterSlotType(position=pos, remaining=count) for pos, count in engine.my_needs().items()]

    @strawberry.field
    def balance(self, info: Info, session_id: int) -> list[CategoryBalanceType]:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
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
        position: str | None = None,
        limit: int = 50,
    ) -> list[DraftBoardRowType]:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        engine = mgr.get_engine(session_id)
        rows = engine.available(position)[:limit]
        return [DraftBoardRowType.from_domain(r) for r in rows]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def start_session(
        self,
        info: Info,
        season: int,
        system: str = "zar",
        version: str = "1.0",
        teams: int | None = None,
        user_team: int = 1,
        format: str = "snake",
        budget: int | None = None,
    ) -> DraftStateType:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        session_id, engine = mgr.start_session(
            season,
            system=system,
            version=version,
            teams=teams,
            user_team=user_team,
            fmt=format,
            budget=budget,
        )
        return DraftStateType.from_state(session_id, engine.state)

    @strawberry.mutation
    def pick(
        self,
        info: Info,
        session_id: int,
        player_id: int,
        position: str,
        price: int | None = None,
        team: int | None = None,
    ) -> PickResultType:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        engine = mgr.get_engine(session_id)

        if team is None:
            if engine.state.config.format == DraftFormat.SNAKE:
                team = engine.team_on_clock()
            else:
                team = engine.state.config.user_team

        draft_pick = mgr.pick(session_id, player_id, team, position, price=price)
        return _build_pick_result(info, session_id, DraftPickType.from_domain(draft_pick))

    @strawberry.mutation
    def undo(self, info: Info, session_id: int) -> PickResultType:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        undone = mgr.undo(session_id)
        return _build_pick_result(info, session_id, DraftPickType.from_domain(undone))

    @strawberry.mutation
    def end_session(self, info: Info, session_id: int) -> bool:
        ctx = _get_context(info)
        mgr = ctx.session_manager
        assert mgr is not None  # noqa: S101
        mgr.end_session(session_id)
        return True
