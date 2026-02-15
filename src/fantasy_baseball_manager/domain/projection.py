from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Projection:
    player_id: int
    season: int
    system: str
    version: str
    player_type: str
    stat_json: dict[str, Any]
    source_type: str = "first_party"
    id: int | None = None
    loaded_at: str | None = None


@dataclass(frozen=True)
class PlayerProjection:
    player_name: str
    system: str
    version: str
    source_type: str
    player_type: str
    stats: dict[str, Any]


@dataclass(frozen=True)
class SystemSummary:
    system: str
    version: str
    source_type: str
    batter_count: int
    pitcher_count: int
