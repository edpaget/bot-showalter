from dataclasses import dataclass, field


@dataclass
class PlayerRates:
    player_id: str
    name: str
    year: int
    age: int
    rates: dict[str, float] = field(default_factory=dict)
    opportunities: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
