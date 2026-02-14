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
