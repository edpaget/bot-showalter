from dataclasses import dataclass


@dataclass(frozen=True)
class LevelFactor:
    level: str
    season: int
    factor: float
    k_factor: float
    bb_factor: float
    iso_factor: float
    babip_factor: float
    id: int | None = None
    loaded_at: str | None = None
