from dataclasses import dataclass


@dataclass(frozen=True)
class PositionAppearance:
    player_id: int
    season: int
    position: str
    games: int
    id: int | None = None
    loaded_at: str | None = None
