from typing import TYPE_CHECKING, Annotated, Any, cast

import strawberry

from fantasy_baseball_manager.domain import Position, position_from_raw

if TYPE_CHECKING:
    from fantasy_baseball_manager.config_toml import WebConfig
    from fantasy_baseball_manager.domain import (
        AdjustedValuation,
        ArbitrageReport,
        CategoryConfig,
        CategoryNeed,
        DraftBoard,
        DraftBoardRow,
        DraftSessionRecord,
        DraftTrade,
        FallingPlayer,
        KeeperDecision,
        KeeperPlanResult,
        KeeperScenarioResult,
        LeagueKeeperOverview,
        LeagueSettings,
        PickTradeEvaluation,
        PickValue,
        PlayerProjection,
        PlayerRecommendation,
        PlayerSummary,
        PlayerTier,
        PlayerValuation,
        PositionScarcity,
        ProjectedKeeper,
        ReachPick,
        Recommendation,
        Roster,
        RosterEntry,
        TeamCategoryProjection,
        TeamKeeperProjection,
        TeamSeasonStats,
        TradeTarget,
        ValueOverADP,
        ValueOverADPReport,
        YahooDraftSetupInfo,
        YahooLeagueInfo,
        YahooTeam,
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
class KeeperInfoType:
    player_id: int
    player_name: str
    position: str
    team_name: str
    cost: float | None
    value: float
    player_type: str | None = None

    @staticmethod
    def from_dict(d: dict[str, object]) -> KeeperInfoType:
        raw_cost = d.get("cost")
        raw_player_type = d.get("player_type")
        return KeeperInfoType(
            player_id=int(str(d["player_id"])),
            player_name=str(d["player_name"]),
            position=str(d["position"]),
            team_name=str(d["team_name"]),
            cost=float(str(raw_cost)) if raw_cost is not None else None,
            value=float(str(d["value"])),
            player_type=str(raw_player_type) if raw_player_type is not None else None,
        )


@strawberry.type
class DraftTradeType:
    team_a: int
    team_b: int
    team_a_gives: list[int]
    team_b_gives: list[int]

    @staticmethod
    def from_domain(trade: DraftTrade) -> DraftTradeType:
        return DraftTradeType(
            team_a=trade.team_a,
            team_b=trade.team_b,
            team_a_gives=list(trade.team_a_gives),
            team_b_gives=list(trade.team_b_gives),
        )


@strawberry.type
class PickValueType:
    pick_number: int
    value: float

    @staticmethod
    def from_domain(pv: PickValue) -> PickValueType:
        return PickValueType(
            pick_number=pv.pick,
            value=pv.expected_value,
        )


@strawberry.type
class PickTradeEvaluationType:
    gives_value: float
    receives_value: float
    net_value: float
    gives_detail: list[PickValueType]
    receives_detail: list[PickValueType]
    recommendation: str

    @staticmethod
    def from_domain(evaluation: PickTradeEvaluation) -> PickTradeEvaluationType:
        return PickTradeEvaluationType(
            gives_value=evaluation.gives_value,
            receives_value=evaluation.receives_value,
            net_value=evaluation.net_value,
            gives_detail=[PickValueType.from_domain(pv) for pv in evaluation.gives_detail],
            receives_detail=[PickValueType.from_domain(pv) for pv in evaluation.receives_detail],
            recommendation=evaluation.recommendation,
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
    keeper_count: int
    trades: list[DraftTradeType]
    team_names: strawberry.scalars.JSON | None = None
    draft_order: list[int] | None = None

    @staticmethod
    def from_state(
        session_id: int,
        state: DraftState,
        *,
        keeper_count: int = 0,
        trades: list[DraftTrade] | None = None,
        team_names: dict[int, str] | None = None,
    ) -> DraftStateType:
        is_auction = state.config.format.value == "auction"
        return DraftStateType(
            session_id=session_id,
            current_pick=state.current_pick,
            picks=[DraftPickType.from_domain(p) for p in state.picks],
            format=state.config.format.value,
            teams=state.config.teams,
            user_team=state.config.user_team,
            budget_remaining=state.team_budgets[state.config.user_team] if is_auction else None,
            keeper_count=keeper_count,
            trades=[DraftTradeType.from_domain(t) for t in (trades or [])],
            team_names=cast("Any", team_names) if team_names else None,
            draft_order=list(state.config.draft_order) if state.config.draft_order else None,
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
class PlayerRecommendationType:
    player_id: int
    player_name: str
    category_impact: float
    tradeoff_categories: list[str]

    @staticmethod
    def from_domain(rec: PlayerRecommendation) -> PlayerRecommendationType:
        return PlayerRecommendationType(
            player_id=rec.player_id,
            player_name=rec.player_name,
            category_impact=rec.category_impact,
            tradeoff_categories=list(rec.tradeoff_categories),
        )


@strawberry.type
class CategoryNeedType:
    category: str
    current_rank: int
    target_rank: int
    best_available: list[PlayerRecommendationType]

    @staticmethod
    def from_domain(need: CategoryNeed) -> CategoryNeedType:
        return CategoryNeedType(
            category=need.category,
            current_rank=need.current_rank,
            target_rank=need.target_rank,
            best_available=[PlayerRecommendationType.from_domain(r) for r in need.best_available],
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
    balance: list[CategoryBalanceType]
    category_needs: list[CategoryNeedType]


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


@strawberry.type
class TradeEvent:
    session_id: int
    trade: DraftTradeType
    action: str  # "trade" or "undo"
    state: DraftStateType


DraftEventType = Annotated[
    PickEvent | UndoEvent | SessionEvent | ArbitrageAlertEvent | TradeEvent,
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
    player_id: int | None = None

    @staticmethod
    def from_domain(proj: PlayerProjection) -> ProjectionType:
        return ProjectionType(
            player_name=proj.player_name,
            system=proj.system,
            version=proj.version,
            source_type=proj.source_type,
            player_type=proj.player_type,
            stats=cast("Any", dict(proj.stats)),
            player_id=proj.player_id,
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
class YahooLeagueInfoType:
    league_key: str
    league_name: str
    season: int
    num_teams: int
    is_keeper: bool
    max_keepers: int | None
    user_team_name: str | None

    @staticmethod
    def from_domain(info: YahooLeagueInfo) -> YahooLeagueInfoType:
        return YahooLeagueInfoType(
            league_key=info.league_key,
            league_name=info.league_name,
            season=info.season,
            num_teams=info.num_teams,
            is_keeper=info.is_keeper,
            max_keepers=info.max_keepers,
            user_team_name=info.user_team_name,
        )


@strawberry.type
class WebConfigType:
    projections: list[SystemVersionType]
    valuations: list[SystemVersionType]
    yahoo_league: YahooLeagueInfoType | None

    @staticmethod
    def from_domain(
        config: WebConfig,
        yahoo_league_info: YahooLeagueInfo | None = None,
    ) -> WebConfigType:
        return WebConfigType(
            projections=[SystemVersionType(system=sv.system, version=sv.version) for sv in config.projections],
            valuations=[SystemVersionType(system=sv.system, version=sv.version) for sv in config.valuations],
            yahoo_league=YahooLeagueInfoType.from_domain(yahoo_league_info) if yahoo_league_info is not None else None,
        )


@strawberry.type
class KeeperDecisionType:
    player_id: int
    player_name: str
    position: Position
    cost: float
    surplus: float
    projected_value: float
    recommendation: str

    @staticmethod
    def from_domain(d: KeeperDecision) -> KeeperDecisionType:
        return KeeperDecisionType(
            player_id=d.player_id,
            player_name=d.player_name,
            position=position_from_raw(d.position),
            cost=d.cost,
            surplus=d.surplus,
            projected_value=d.projected_value,
            recommendation=d.recommendation,
        )


@strawberry.type
class AdjustedValuationType:
    player_id: int
    player_name: str
    player_type: str
    position: Position
    original_value: float
    adjusted_value: float
    value_change: float

    @staticmethod
    def from_domain(av: AdjustedValuation) -> AdjustedValuationType:
        return AdjustedValuationType(
            player_id=av.player_id,
            player_name=av.player_name,
            player_type=av.player_type,
            position=position_from_raw(av.position),
            original_value=av.original_value,
            adjusted_value=av.adjusted_value,
            value_change=av.value_change,
        )


@strawberry.type
class KeeperScenarioType:
    keeper_ids: list[int]
    keepers: list[KeeperDecisionType]
    total_surplus: float
    board_preview: list[AdjustedValuationType]
    scarcity: list[PositionScarcityType]
    category_needs: list[CategoryNeedType]
    strongest_categories: list[str]
    weakest_categories: list[str]

    @staticmethod
    def from_domain(s: KeeperScenarioResult) -> KeeperScenarioType:
        return KeeperScenarioType(
            keeper_ids=sorted(s.keeper_ids),
            keepers=[KeeperDecisionType.from_domain(d) for d in s.keeper_decisions],
            total_surplus=s.total_surplus,
            board_preview=[AdjustedValuationType.from_domain(a) for a in s.board_preview],
            scarcity=[PositionScarcityType.from_domain(sc) for sc in s.scarcity],
            category_needs=[CategoryNeedType.from_domain(n) for n in s.category_needs],
            strongest_categories=list(s.strongest_categories),
            weakest_categories=list(s.weakest_categories),
        )


@strawberry.type
class KeeperPlanType:
    scenarios: list[KeeperScenarioType]

    @staticmethod
    def from_domain(plan: KeeperPlanResult) -> KeeperPlanType:
        return KeeperPlanType(
            scenarios=[KeeperScenarioType.from_domain(s) for s in plan.scenarios],
        )


@strawberry.type
class YahooTeamType:
    team_key: str
    name: str
    manager_name: str
    is_owned_by_user: bool

    @staticmethod
    def from_domain(team: YahooTeam) -> YahooTeamType:
        return YahooTeamType(
            team_key=team.team_key,
            name=team.name,
            manager_name=team.manager_name,
            is_owned_by_user=team.is_owned_by_user,
        )


@strawberry.type
class YahooStandingsEntryType:
    team_key: str
    team_name: str
    final_rank: int
    stat_values: strawberry.scalars.JSON

    @staticmethod
    def from_domain(stats: TeamSeasonStats) -> YahooStandingsEntryType:
        return YahooStandingsEntryType(
            team_key=stats.team_key,
            team_name=stats.team_name,
            final_rank=stats.final_rank,
            stat_values=cast("Any", dict(stats.stat_values)),
        )


@strawberry.type
class YahooRosterEntryType:
    yahoo_player_key: str
    player_name: str
    position: str
    acquisition_type: str
    player_id: int | None

    @staticmethod
    def from_domain(entry: RosterEntry) -> YahooRosterEntryType:
        return YahooRosterEntryType(
            yahoo_player_key=entry.yahoo_player_key,
            player_name=entry.player_name,
            position=entry.position,
            acquisition_type=entry.acquisition_type,
            player_id=entry.player_id,
        )


@strawberry.type
class YahooRosterType:
    team_key: str
    league_key: str
    season: int
    week: int
    as_of: str
    entries: list[YahooRosterEntryType]

    @staticmethod
    def from_domain(roster: Roster) -> YahooRosterType:
        return YahooRosterType(
            team_key=roster.team_key,
            league_key=roster.league_key,
            season=roster.season,
            week=roster.week,
            as_of=roster.as_of.isoformat(),
            entries=[YahooRosterEntryType.from_domain(e) for e in roster.entries],
        )


@strawberry.type
class YahooDraftSetupInfoType:
    num_teams: int
    draft_format: str
    user_team_id: int
    team_names: strawberry.scalars.JSON
    draft_order: list[int]
    is_keeper: bool
    max_keepers: int | None
    keeper_player_ids: strawberry.scalars.JSON  # [[player_id, player_type], ...]

    @staticmethod
    def from_domain(info: YahooDraftSetupInfo) -> YahooDraftSetupInfoType:
        return YahooDraftSetupInfoType(
            num_teams=info.num_teams,
            draft_format=info.draft_format,
            user_team_id=info.user_team_id,
            team_names=cast("Any", info.team_names),
            draft_order=list(info.draft_order),
            is_keeper=info.is_keeper,
            max_keepers=info.max_keepers,
            keeper_player_ids=cast("Any", list(info.keeper_player_ids)),
        )


@strawberry.type
class ProjectedKeeperType:
    player_id: int
    player_name: str
    position: str
    value: float
    category_scores: strawberry.scalars.JSON

    @staticmethod
    def from_domain(pk: ProjectedKeeper) -> ProjectedKeeperType:
        return ProjectedKeeperType(
            player_id=pk.player_id,
            player_name=pk.player_name,
            position=pk.position,
            value=pk.value,
            category_scores=cast("Any", dict(pk.category_scores)),
        )


@strawberry.type
class TeamKeeperProjectionType:
    team_key: str
    team_name: str
    is_user: bool
    keepers: list[ProjectedKeeperType]
    total_value: float
    category_totals: strawberry.scalars.JSON

    @staticmethod
    def from_domain(proj: TeamKeeperProjection) -> TeamKeeperProjectionType:
        return TeamKeeperProjectionType(
            team_key=proj.team_key,
            team_name=proj.team_name,
            is_user=proj.is_user,
            keepers=[ProjectedKeeperType.from_domain(k) for k in proj.keepers],
            total_value=proj.total_value,
            category_totals=cast("Any", dict(proj.category_totals)),
        )


@strawberry.type
class TradeTargetType:
    player_id: int
    player_name: str
    position: str
    value: float
    owning_team_name: str
    owning_team_key: str
    rank_on_team: int

    @staticmethod
    def from_domain(target: TradeTarget) -> TradeTargetType:
        return TradeTargetType(
            player_id=target.player_id,
            player_name=target.player_name,
            position=target.position,
            value=target.value,
            owning_team_name=target.owning_team_name,
            owning_team_key=target.owning_team_key,
            rank_on_team=target.rank_on_team,
        )


@strawberry.type
class LeagueKeeperOverviewType:
    team_projections: list[TeamKeeperProjectionType]
    trade_targets: list[TradeTargetType]
    category_names: list[str]

    @staticmethod
    def from_domain(overview: LeagueKeeperOverview) -> LeagueKeeperOverviewType:
        return LeagueKeeperOverviewType(
            team_projections=[TeamKeeperProjectionType.from_domain(p) for p in overview.team_projections],
            trade_targets=[TradeTargetType.from_domain(t) for t in overview.trade_targets],
            category_names=list(overview.category_names),
        )
