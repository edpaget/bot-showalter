from typing import TYPE_CHECKING, Annotated, Any, cast

import strawberry

from fantasy_baseball_manager.domain import Position, position_from_raw

if TYPE_CHECKING:
    from fantasy_baseball_manager.config_toml import WebConfig
    from fantasy_baseball_manager.domain import (
        ArbitrageReport,
        CategoryConfig,
        DraftBoard,
        DraftBoardRow,
        DraftSessionRecord,
        FallingPlayer,
        LeagueSettings,
        PlayerProjection,
        PlayerSummary,
        PlayerTier,
        PlayerValuation,
        PositionScarcity,
        ReachPick,
        Recommendation,
        TeamCategoryProjection,
        ValueOverADP,
        ValueOverADPReport,
    )
    from fantasy_baseball_manager.services import DraftPick, DraftState

PositionGQL = strawberry.enum(Position, name="Position")


@strawberry.type
class DraftBoardRowType:
    player_id: int
    player_name: str
    rank: int
    player_type: str
    position: Position
    value: float
    category_z_scores: strawberry.scalars.JSON
    age: int | None
    bats_throws: str | None
    tier: int | None
    adp_overall: float | None
    adp_rank: int | None
    adp_delta: int | None
    breakout_rank: int | None
    bust_rank: int | None

    @staticmethod
    def from_domain(row: DraftBoardRow) -> DraftBoardRowType:
        return DraftBoardRowType(
            player_id=row.player_id,
            player_name=row.player_name,
            rank=row.rank,
            player_type=row.player_type,
            position=position_from_raw(row.position),
            value=row.value,
            category_z_scores=cast("Any", dict(row.category_z_scores)),
            age=row.age,
            bats_throws=row.bats_throws,
            tier=row.tier,
            adp_overall=row.adp_overall,
            adp_rank=row.adp_rank,
            adp_delta=row.adp_delta,
            breakout_rank=row.breakout_rank,
            bust_rank=row.bust_rank,
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
    position: Position
    tier: int
    value: float
    rank: int

    @staticmethod
    def from_domain(pt: PlayerTier) -> PlayerTierType:
        return PlayerTierType(
            player_id=pt.player_id,
            player_name=pt.player_name,
            position=position_from_raw(pt.position),
            tier=pt.tier,
            value=pt.value,
            rank=pt.rank,
        )


@strawberry.type
class PositionScarcityType:
    position: Position
    tier_1_value: float
    replacement_value: float
    total_surplus: float
    dropoff_slope: float
    steep_rank: int | None

    @staticmethod
    def from_domain(ps: PositionScarcity) -> PositionScarcityType:
        return PositionScarcityType(
            position=position_from_raw(ps.position),
            tier_1_value=ps.tier_1_value,
            replacement_value=ps.replacement_value,
            total_surplus=ps.total_surplus,
            dropoff_slope=ps.dropoff_slope,
            steep_rank=ps.steep_rank,
        )


@strawberry.type
class DraftPickType:
    pick_number: int
    team: int
    player_id: int
    player_name: str
    position: Position
    price: int | None

    @staticmethod
    def from_domain(pick: DraftPick) -> DraftPickType:
        return DraftPickType(
            pick_number=pick.pick_number,
            team=pick.team,
            player_id=pick.player_id,
            player_name=pick.player_name,
            position=position_from_raw(pick.position),
            price=pick.price,
        )


@strawberry.type
class DraftSessionSummaryType:
    id: int
    league: str
    season: int
    teams: int
    format: str
    user_team: int
    status: str
    pick_count: int
    created_at: str
    updated_at: str
    system: str
    version: str

    @staticmethod
    def from_domain(record: DraftSessionRecord, pick_count: int) -> DraftSessionSummaryType:
        return DraftSessionSummaryType(
            id=record.id or 0,
            league=record.league,
            season=record.season,
            teams=record.teams,
            format=record.format,
            user_team=record.user_team,
            status=record.status,
            pick_count=pick_count,
            created_at=record.created_at,
            updated_at=record.updated_at,
            system=record.system,
            version=record.version,
        )


@strawberry.type
class DraftStateType:
    session_id: int
    current_pick: int
    picks: list[DraftPickType]
    format: str
    teams: int
    user_team: int
    budget_remaining: int | None

    @staticmethod
    def from_state(session_id: int, state: DraftState) -> DraftStateType:
        is_auction = state.config.format.value == "auction"
        return DraftStateType(
            session_id=session_id,
            current_pick=state.current_pick,
            picks=[DraftPickType.from_domain(p) for p in state.picks],
            format=state.config.format.value,
            teams=state.config.teams,
            user_team=state.config.user_team,
            budget_remaining=state.team_budgets[state.config.user_team] if is_auction else None,
        )


@strawberry.type
class RecommendationType:
    player_id: int
    player_name: str
    position: Position
    value: float
    score: float
    reason: str

    @staticmethod
    def from_domain(rec: Recommendation) -> RecommendationType:
        return RecommendationType(
            player_id=rec.player_id,
            player_name=rec.player_name,
            position=position_from_raw(rec.position),
            value=rec.value,
            score=rec.score,
            reason=rec.reason,
        )


@strawberry.type
class RosterSlotType:
    position: Position
    remaining: int


@strawberry.type
class CategoryBalanceType:
    category: str
    projected_value: float
    league_rank_estimate: int
    strength: str

    @staticmethod
    def from_domain(proj: TeamCategoryProjection) -> CategoryBalanceType:
        return CategoryBalanceType(
            category=proj.category,
            projected_value=proj.projected_value,
            league_rank_estimate=proj.league_rank_estimate,
            strength=proj.strength,
        )


@strawberry.type
class FallingPlayerType:
    player_id: int
    player_name: str
    position: str
    adp: float
    current_pick: int
    picks_past_adp: float
    value: float
    value_rank: int
    arbitrage_score: float

    @staticmethod
    def from_domain(fp: FallingPlayer) -> FallingPlayerType:
        return FallingPlayerType(
            player_id=fp.player_id,
            player_name=fp.player_name,
            position=fp.position,
            adp=fp.adp,
            current_pick=fp.current_pick,
            picks_past_adp=fp.picks_past_adp,
            value=fp.value,
            value_rank=fp.value_rank,
            arbitrage_score=fp.arbitrage_score,
        )


@strawberry.type
class ReachPickType:
    player_id: int
    player_name: str
    position: str
    adp: float
    pick_number: int
    picks_ahead_of_adp: float
    drafter_team: int

    @staticmethod
    def from_domain(rp: ReachPick) -> ReachPickType:
        return ReachPickType(
            player_id=rp.player_id,
            player_name=rp.player_name,
            position=rp.position,
            adp=rp.adp,
            pick_number=rp.pick_number,
            picks_ahead_of_adp=rp.picks_ahead_of_adp,
            drafter_team=rp.drafter_team,
        )


@strawberry.type
class ArbitrageReportType:
    current_pick: int
    falling: list[FallingPlayerType]
    reaches: list[ReachPickType]

    @staticmethod
    def from_domain(report: ArbitrageReport) -> ArbitrageReportType:
        return ArbitrageReportType(
            current_pick=report.current_pick,
            falling=[FallingPlayerType.from_domain(f) for f in report.falling],
            reaches=[ReachPickType.from_domain(r) for r in report.reaches],
        )


@strawberry.type
class ArbitrageAlertEvent:
    session_id: int
    falling: list[FallingPlayerType]


@strawberry.type
class PickResultType:
    pick: DraftPickType
    state: DraftStateType
    recommendations: list[RecommendationType]
    roster: list[DraftPickType]
    needs: list[RosterSlotType]
    arbitrage: ArbitrageReportType | None


@strawberry.type
class PickEvent:
    pick: DraftPickType
    session_id: int


@strawberry.type
class UndoEvent:
    pick: DraftPickType
    session_id: int


@strawberry.type
class SessionEvent:
    session_id: int
    event_type: str


DraftEventType = Annotated[
    PickEvent | UndoEvent | SessionEvent | ArbitrageAlertEvent,
    strawberry.union("DraftEventType"),
]


@strawberry.type
class YahooPollStatusType:
    active: bool
    last_poll_at: str | None
    picks_ingested: int


@strawberry.type
class ProjectionType:
    player_name: str
    system: str
    version: str
    source_type: str
    player_type: str
    stats: strawberry.scalars.JSON

    @staticmethod
    def from_domain(proj: PlayerProjection) -> ProjectionType:
        return ProjectionType(
            player_name=proj.player_name,
            system=proj.system,
            version=proj.version,
            source_type=proj.source_type,
            player_type=proj.player_type,
            stats=cast("Any", dict(proj.stats)),
        )


@strawberry.type
class ValuationType:
    player_name: str
    system: str
    version: str
    projection_system: str
    projection_version: str
    player_type: str
    position: Position
    value: float
    rank: int
    category_scores: strawberry.scalars.JSON

    @staticmethod
    def from_domain(val: PlayerValuation) -> ValuationType:
        return ValuationType(
            player_name=val.player_name,
            system=val.system,
            version=val.version,
            projection_system=val.projection_system,
            projection_version=val.projection_version,
            player_type=val.player_type,
            position=position_from_raw(val.position),
            value=val.value,
            rank=val.rank,
            category_scores=cast("Any", dict(val.category_scores)),
        )


@strawberry.type
class ADPReportRowType:
    player_id: int
    player_name: str
    player_type: str
    position: Position
    zar_rank: int
    zar_value: float
    adp_rank: int
    adp_pick: float
    rank_delta: int
    provider: str

    @staticmethod
    def from_domain(row: ValueOverADP) -> ADPReportRowType:
        return ADPReportRowType(
            player_id=row.player_id,
            player_name=row.player_name,
            player_type=row.player_type,
            position=position_from_raw(row.position),
            zar_rank=row.zar_rank,
            zar_value=row.zar_value,
            adp_rank=row.adp_rank,
            adp_pick=row.adp_pick,
            rank_delta=row.rank_delta,
            provider=row.provider,
        )


@strawberry.type
class ADPReportType:
    season: int
    system: str
    version: str
    provider: str
    buy_targets: list[ADPReportRowType]
    avoid_list: list[ADPReportRowType]
    unranked_valuable: list[ADPReportRowType]
    n_matched: int

    @staticmethod
    def from_domain(report: ValueOverADPReport) -> ADPReportType:
        return ADPReportType(
            season=report.season,
            system=report.system,
            version=report.version,
            provider=report.provider,
            buy_targets=[ADPReportRowType.from_domain(r) for r in report.buy_targets],
            avoid_list=[ADPReportRowType.from_domain(r) for r in report.avoid_list],
            unranked_valuable=[ADPReportRowType.from_domain(r) for r in report.unranked_valuable],
            n_matched=report.n_matched,
        )


@strawberry.type
class PlayerSummaryType:
    player_id: int
    name: str
    team: str
    age: int | None
    primary_position: str
    bats: str | None
    throws: str | None
    experience: int

    @staticmethod
    def from_domain(summary: PlayerSummary) -> PlayerSummaryType:
        return PlayerSummaryType(
            player_id=summary.player_id,
            name=summary.name,
            team=summary.team,
            age=summary.age,
            primary_position=summary.primary_position,
            bats=summary.bats,
            throws=summary.throws,
            experience=summary.experience,
        )


@strawberry.type
class SystemVersionType:
    system: str
    version: str


@strawberry.type
class WebConfigType:
    projections: list[SystemVersionType]
    valuations: list[SystemVersionType]

    @staticmethod
    def from_domain(config: WebConfig) -> WebConfigType:
        return WebConfigType(
            projections=[SystemVersionType(system=sv.system, version=sv.version) for sv in config.projections],
            valuations=[SystemVersionType(system=sv.system, version=sv.version) for sv in config.valuations],
        )
