from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.category_tracker import CategoryNeed
    from fantasy_baseball_manager.domain.positional_scarcity import PositionScarcity


@dataclass(frozen=True)
class KeeperCost:
    player_id: int
    season: int
    league: str
    cost: float
    source: str
    years_remaining: int = 1
    original_round: int | None = None
    id: int | None = None
    loaded_at: str | None = None


@dataclass(frozen=True)
class KeeperDecision:
    player_id: int
    player_name: str
    position: str
    cost: float
    projected_value: float
    surplus: float
    years_remaining: int
    recommendation: str  # "keep" | "release"
    original_round: int | None = None


@dataclass(frozen=True)
class TradePlayerDetail:
    player_id: int
    player_name: str
    position: str
    cost: float
    projected_value: float
    surplus: float
    years_remaining: int


@dataclass(frozen=True)
class TradeEvaluation:
    team_a_gives: list[TradePlayerDetail]
    team_b_gives: list[TradePlayerDetail]
    team_a_surplus_delta: float
    team_b_surplus_delta: float
    winner: str  # "team_a" | "team_b" | "even"


@dataclass(frozen=True)
class ProjectedKeeper:
    player_id: int
    player_name: str
    position: str
    value: float
    category_scores: dict[str, float]


@dataclass(frozen=True)
class TeamKeeperProjection:
    team_key: str
    team_name: str
    is_user: bool
    keepers: tuple[ProjectedKeeper, ...]  # sorted by value desc
    total_value: float
    category_totals: dict[str, float]


@dataclass(frozen=True)
class TradeTarget:
    player_id: int
    player_name: str
    position: str
    value: float
    owning_team_name: str
    owning_team_key: str
    rank_on_team: int


@dataclass(frozen=True)
class LeagueKeeperOverview:
    team_projections: tuple[TeamKeeperProjection, ...]  # sorted by total_value desc
    trade_targets: tuple[TradeTarget, ...]  # sorted by value desc
    category_names: tuple[str, ...]


@dataclass(frozen=True)
class LeagueKeeper:
    player_id: int
    season: int
    league: str
    team_name: str
    cost: float | None = None
    source: str | None = None
    player_type: str | None = None
    id: int | None = None


@dataclass(frozen=True)
class KeeperScenarioResult:
    keeper_ids: frozenset[int]
    keeper_decisions: tuple[KeeperDecision, ...]
    total_surplus: float
    board_preview: tuple[AdjustedValuation, ...]
    scarcity: tuple[PositionScarcity, ...]
    category_needs: tuple[CategoryNeed, ...]
    strongest_categories: tuple[str, ...]
    weakest_categories: tuple[str, ...]


@dataclass(frozen=True)
class KeeperPlanResult:
    scenarios: tuple[KeeperScenarioResult, ...]


@dataclass(frozen=True)
class AdjustedValuation:
    player_id: int
    player_name: str
    player_type: str
    position: str
    original_value: float
    adjusted_value: float
    value_change: float
