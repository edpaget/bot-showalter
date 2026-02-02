from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.draft.simulation_models import SimulationPick
    from fantasy_baseball_manager.valuation.models import PlayerValue

_PITCHER_POSITIONS: frozenset[str] = frozenset({"SP", "RP", "P"})


@dataclass(frozen=True)
class PositionNotBeforeRound:
    """Veto picks of a specific position before a given round."""

    position: str
    earliest_round: int

    @property
    def name(self) -> str:
        return f"no_{self.position}_before_round_{self.earliest_round}"

    def evaluate(
        self,
        player: PlayerValue,
        eligible_positions: tuple[str, ...],
        round_number: int,
        total_rounds: int,
        picks_so_far: list[SimulationPick],
    ) -> float:
        if self.position not in eligible_positions:
            return 1.0
        if round_number < self.earliest_round:
            return 0.0
        return 1.0


@dataclass(frozen=True)
class MaxPositionCount:
    """Veto picks when already at max count for a position."""

    position: str
    max_count: int

    @property
    def name(self) -> str:
        return f"max_{self.max_count}_{self.position}"

    def evaluate(
        self,
        player: PlayerValue,
        eligible_positions: tuple[str, ...],
        round_number: int,
        total_rounds: int,
        picks_so_far: list[SimulationPick],
    ) -> float:
        if self.position not in eligible_positions:
            return 1.0
        count = sum(1 for p in picks_so_far if p.position == self.position)
        if count >= self.max_count:
            return 0.0
        return 1.0


@dataclass(frozen=True)
class PitcherBatterRatio:
    """Penalize pitcher picks when pitcher fraction exceeds max."""

    max_pitcher_fraction: float

    @property
    def name(self) -> str:
        return f"pitcher_ratio_max_{self.max_pitcher_fraction}"

    def evaluate(
        self,
        player: PlayerValue,
        eligible_positions: tuple[str, ...],
        round_number: int,
        total_rounds: int,
        picks_so_far: list[SimulationPick],
    ) -> float:
        if player.position_type != "P":
            return 1.0
        if not picks_so_far:
            return 1.0
        pitcher_count = sum(1 for p in picks_so_far if p.position in _PITCHER_POSITIONS)
        total = len(picks_so_far)
        current_fraction = pitcher_count / total
        if current_fraction > self.max_pitcher_fraction:
            overshoot = current_fraction - self.max_pitcher_fraction
            return max(0.3, 1.0 - overshoot * 3)
        return 1.0
