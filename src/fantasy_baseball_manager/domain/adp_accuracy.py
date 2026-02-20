from dataclasses import dataclass


@dataclass(frozen=True)
class ADPAccuracyPlayer:
    player_id: int
    player_name: str
    player_type: str
    adp_rank: int
    actual_rank: int
    actual_value: float
    implied_value: float
    value_error: float


@dataclass(frozen=True)
class ADPAccuracyResult:
    season: int
    provider: str
    rank_correlation: float
    value_rmse: float
    value_mae: float
    top_n_precision: dict[int, float]
    n_matched: int
    players: list[ADPAccuracyPlayer]


@dataclass(frozen=True)
class SystemAccuracyResult:
    system: str
    version: str
    season: int
    rank_correlation: float
    value_rmse: float
    value_mae: float
    top_n_precision: dict[int, float]
    n_matched: int


@dataclass(frozen=True)
class ADPAccuracyReport:
    provider: str
    seasons: list[int]
    adp_results: list[ADPAccuracyResult]
    comparison: list[SystemAccuracyResult] | None
    mean_rank_correlation: float
    mean_value_rmse: float
    mean_top_n_precision: dict[int, float]
