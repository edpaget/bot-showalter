from dataclasses import dataclass


@dataclass(frozen=True)
class Valuation:
    player_id: int
    season: int
    system: str
    version: str
    projection_system: str
    projection_version: str
    player_type: str
    position: str
    value: float
    rank: int
    category_scores: dict[str, float]
    id: int | None = None
    loaded_at: str | None = None
