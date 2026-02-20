from dataclasses import dataclass


@dataclass(frozen=True)
class ADP:
    player_id: int
    season: int
    provider: str
    overall_pick: float
    rank: int
    positions: str
    as_of: str | None = None
    id: int | None = None
    loaded_at: str | None = None
