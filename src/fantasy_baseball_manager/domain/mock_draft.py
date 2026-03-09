from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BotStrategy(StrEnum):
    ADP_BASED = "adp_based"
    BEST_VALUE = "best_value"
    POSITIONAL_NEED = "positional_need"
    RANDOM = "random"


@dataclass(frozen=True)
class DraftPick:
    round: int
    pick: int
    team_idx: int
    player_id: int
    player_name: str
    position: str
    value: float


@dataclass(frozen=True)
class DraftResult:
    picks: list[DraftPick]
    rosters: dict[int, list[DraftPick]]
    snake: bool


@dataclass(frozen=True)
class SimulationSummary:
    n_simulations: int
    team_idx: int | None
    avg_roster_value: float
    median_roster_value: float
    p10_roster_value: float
    p25_roster_value: float
    p75_roster_value: float
    p90_roster_value: float


@dataclass(frozen=True)
class PlayerDraftFrequency:
    player_id: int
    player_name: str
    pct_drafted: float
    avg_round_drafted: float
    avg_pick_drafted: float


@dataclass(frozen=True)
class StrategyComparison:
    strategy_name: str
    avg_value: float
    win_rate: float


@dataclass(frozen=True)
class BatchSimulationResult:
    summary: SimulationSummary
    player_frequencies: list[PlayerDraftFrequency]
    strategy_comparisons: list[StrategyComparison]
    user_rosters: list[list[DraftPick]]
    user_roster_values: list[float]
