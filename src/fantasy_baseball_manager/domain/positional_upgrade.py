from dataclasses import dataclass, field


@dataclass(frozen=True)
class RosterSlot:
    position: str
    player_id: int | None = None
    player_name: str | None = None
    value: float = 0.0
    category_z_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RosterState:
    slots: list[RosterSlot]
    open_positions: list[str]
    total_value: float
    category_totals: dict[str, float]


@dataclass(frozen=True)
class PositionUpgrade:
    position: str
    current_player: str | None
    current_value: float
    best_available: str
    best_available_value: float
    upgrade_value: float
    next_best: str | None
    dropoff_to_next: float
    urgency: str


@dataclass(frozen=True)
class OpportunityCost:
    position: str
    recommended_player: str
    marginal_value: float
    opportunity_cost: float
    net_value: float
    recommendation: str


@dataclass(frozen=True)
class MarginalValue:
    player_id: int
    player_name: str
    position: str
    raw_value: float
    marginal_value: float
    category_impacts: dict[str, float]
    fills_need: bool
    upgrade_over: str | None = None
