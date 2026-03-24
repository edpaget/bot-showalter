from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class StatDistribution:
    stat: str
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    mean: float | None = None
    std: float | None = None
    family: str | None = None


@dataclass(frozen=True)
class Projection:
    player_id: int
    season: int
    system: str
    version: str
    player_type: PlayerType
    stat_json: dict[str, Any]
    source_type: str = "first_party"
    id: int | None = None
    loaded_at: str | None = None
    distributions: dict[str, StatDistribution] | None = None


@dataclass(frozen=True)
class PlayerProjection:
    player_name: str
    system: str
    version: str
    source_type: str
    player_type: PlayerType
    stats: dict[str, Any]
    player_id: int | None = None


@dataclass(frozen=True)
class PTSourceSummary:
    system: str
    version: str
    batter_count: int  # players with pa > 0
    pitcher_count: int  # players with ip > 0


@dataclass(frozen=True)
class SystemSummary:
    system: str
    version: str
    source_type: str
    batter_count: int
    pitcher_count: int
