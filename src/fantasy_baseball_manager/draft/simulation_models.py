from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fantasy_baseball_manager.draft.models import RosterConfig
    from fantasy_baseball_manager.valuation.models import PlayerValue, StatCategory


@runtime_checkable
class DraftRule(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        player: PlayerValue,
        eligible_positions: tuple[str, ...],
        round_number: int,
        total_rounds: int,
        picks_so_far: list[SimulationPick],
    ) -> float: ...


@dataclass(frozen=True)
class DraftStrategy:
    name: str
    category_weights: dict[StatCategory, float]
    rules: tuple[DraftRule, ...]
    noise_scale: float = 0.15

    def __post_init__(self) -> None:
        if self.noise_scale < 0:
            msg = "noise_scale must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class TeamConfig:
    team_id: int
    name: str
    strategy: DraftStrategy
    keepers: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimulationConfig:
    teams: tuple[TeamConfig, ...]
    roster_config: RosterConfig
    total_rounds: int
    seed: int | None = None


@dataclass(frozen=True)
class SimulationPick:
    overall_pick: int
    round_number: int
    pick_in_round: int
    team_id: int
    team_name: str
    player_id: str
    player_name: str
    position: str | None
    adjusted_value: float


@dataclass(frozen=True)
class TeamResult:
    team_id: int
    team_name: str
    strategy_name: str
    picks: tuple[SimulationPick, ...]
    category_totals: dict[StatCategory, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    pick_log: tuple[SimulationPick, ...]
    team_results: tuple[TeamResult, ...]
    config: SimulationConfig
