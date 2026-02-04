from dataclasses import dataclass

from fantasy_baseball_manager.valuation.models import StatCategory


@dataclass(frozen=True)
class StatAccuracy:
    category: StatCategory
    sample_size: int
    rmse: float
    mae: float
    correlation: float


@dataclass(frozen=True)
class RankAccuracy:
    sample_size: int
    spearman_rho: float
    top_n: int
    top_n_precision: float


@dataclass(frozen=True)
class StratumAccuracy:
    """Accuracy metrics for a player segment."""

    stratum_name: str
    sample_size: int
    stat_accuracy: tuple[StatAccuracy, ...]
    rank_accuracy: RankAccuracy | None


@dataclass(frozen=True)
class PlayerResidual:
    """Per-player prediction error."""

    player_id: str
    player_name: str
    category: StatCategory
    projected: float
    actual: float
    residual: float
    abs_residual: float


@dataclass(frozen=True)
class HeadToHeadResult:
    """Comparison of two sources on same players."""

    source_a: str
    source_b: str
    category: StatCategory
    sample_size: int
    a_wins: int
    b_wins: int
    ties: int
    a_win_pct: float
    mean_improvement: float


@dataclass(frozen=True)
class SourceEvaluation:
    source_name: str
    year: int
    batting_stat_accuracy: tuple[StatAccuracy, ...]
    pitching_stat_accuracy: tuple[StatAccuracy, ...]
    batting_rank_accuracy: RankAccuracy | None
    pitching_rank_accuracy: RankAccuracy | None
    batting_strata: tuple[StratumAccuracy, ...] = ()
    pitching_strata: tuple[StratumAccuracy, ...] = ()
    batting_residuals: tuple[PlayerResidual, ...] | None = None
    pitching_residuals: tuple[PlayerResidual, ...] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    evaluations: tuple[SourceEvaluation, ...]
