from typing import TYPE_CHECKING

import strawberry
from strawberry.types import Info  # noqa: TC002 — Strawberry needs this at runtime

from fantasy_baseball_manager.domain import DraftBoard
from fantasy_baseball_manager.services import build_draft_board, compute_scarcity, generate_tiers
from fantasy_baseball_manager.web.types import (
    DraftBoardType,
    LeagueSettingsType,
    PlayerTierType,
    PositionScarcityType,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.web.app import AppContext


def _get_context(info: Info) -> AppContext:
    return info.context["app_context"]


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
