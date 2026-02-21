from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerTier:
    player_id: int
    player_name: str
    position: str
    tier: int
    value: float
    rank: int  # within-position rank (1 = best)


@dataclass(frozen=True)
class TierSummaryEntry:
    position: str
    tier: int
    count: int
    total_value: float
    avg_value: float
    best_player: str


@dataclass(frozen=True)
class TierSummaryReport:
    positions: list[str]  # sorted position names
    max_tier: int  # highest tier number across all positions
    entries: list[TierSummaryEntry]  # flat list, grouped by position then tier
