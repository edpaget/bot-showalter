from typing import TYPE_CHECKING, Any, cast

import strawberry

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        CategoryConfig,
        DraftBoard,
        DraftBoardRow,
        LeagueSettings,
        PlayerTier,
        PositionScarcity,
    )


@strawberry.type
class DraftBoardRowType:
    player_id: int
    player_name: str
    rank: int
    player_type: str
    position: str
    value: float
    category_z_scores: strawberry.scalars.JSON
    age: int | None
    bats_throws: str | None
    tier: int | None
    adp_overall: float | None
    adp_rank: int | None
    adp_delta: int | None

    @staticmethod
    def from_domain(row: DraftBoardRow) -> DraftBoardRowType:
        return DraftBoardRowType(
            player_id=row.player_id,
            player_name=row.player_name,
            rank=row.rank,
            player_type=row.player_type,
            position=row.position,
            value=row.value,
            category_z_scores=cast("Any", dict(row.category_z_scores)),
            age=row.age,
            bats_throws=row.bats_throws,
            tier=row.tier,
            adp_overall=row.adp_overall,
            adp_rank=row.adp_rank,
            adp_delta=row.adp_delta,
        )


@strawberry.type
class DraftBoardType:
    rows: list[DraftBoardRowType]
    batting_categories: list[str]
    pitching_categories: list[str]

    @staticmethod
    def from_domain(board: DraftBoard) -> DraftBoardType:
        return DraftBoardType(
            rows=[DraftBoardRowType.from_domain(r) for r in board.rows],
            batting_categories=list(board.batting_categories),
            pitching_categories=list(board.pitching_categories),
        )


@strawberry.type
class CategoryConfigType:
    key: str
    name: str
    stat_type: str
    direction: str

    @staticmethod
    def from_domain(cat: CategoryConfig) -> CategoryConfigType:
        return CategoryConfigType(
            key=cat.key,
            name=cat.name,
            stat_type=cat.stat_type.value,
            direction=cat.direction.value,
        )


@strawberry.type
class LeagueSettingsType:
    name: str
    format: str
    teams: int
    budget: int
    roster_batters: int
    roster_pitchers: int
    roster_util: int
    batting_categories: list[CategoryConfigType]
    pitching_categories: list[CategoryConfigType]
    positions: strawberry.scalars.JSON
    pitcher_positions: strawberry.scalars.JSON

    @staticmethod
    def from_domain(league: LeagueSettings) -> LeagueSettingsType:
        return LeagueSettingsType(
            name=league.name,
            format=league.format.value,
            teams=league.teams,
            budget=league.budget,
            roster_batters=league.roster_batters,
            roster_pitchers=league.roster_pitchers,
            roster_util=league.roster_util,
            batting_categories=[CategoryConfigType.from_domain(c) for c in league.batting_categories],
            pitching_categories=[CategoryConfigType.from_domain(c) for c in league.pitching_categories],
            positions=cast("Any", dict(league.positions)),
            pitcher_positions=cast("Any", dict(league.pitcher_positions)),
        )


@strawberry.type
class PlayerTierType:
    player_id: int
    player_name: str
    position: str
    tier: int
    value: float
    rank: int

    @staticmethod
    def from_domain(pt: PlayerTier) -> PlayerTierType:
        return PlayerTierType(
            player_id=pt.player_id,
            player_name=pt.player_name,
            position=pt.position,
            tier=pt.tier,
            value=pt.value,
            rank=pt.rank,
        )


@strawberry.type
class PositionScarcityType:
    position: str
    tier_1_value: float
    replacement_value: float
    total_surplus: float
    dropoff_slope: float
    steep_rank: int | None

    @staticmethod
    def from_domain(ps: PositionScarcity) -> PositionScarcityType:
        return PositionScarcityType(
            position=ps.position,
            tier_1_value=ps.tier_1_value,
            replacement_value=ps.replacement_value,
            total_surplus=ps.total_surplus,
            dropoff_slope=ps.dropoff_slope,
            steep_rank=ps.steep_rank,
        )
