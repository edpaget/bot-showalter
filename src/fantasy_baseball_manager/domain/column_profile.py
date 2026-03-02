from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnProfile:
    column: str
    season: int
    player_type: str  # "batter" or "pitcher"
    count: int  # number of player-seasons with non-null values
    null_count: int  # number of player-seasons with null aggregated values
    null_pct: float  # null_count / (count + null_count) * 100
    mean: float
    median: float
    std: float
    min: float
    max: float
    p10: float
    p25: float
    p75: float
    p90: float
    skewness: float
