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
class SourceEvaluation:
    source_name: str
    year: int
    batting_stat_accuracy: tuple[StatAccuracy, ...]
    pitching_stat_accuracy: tuple[StatAccuracy, ...]
    batting_rank_accuracy: RankAccuracy | None
    pitching_rank_accuracy: RankAccuracy | None


@dataclass(frozen=True)
class EvaluationResult:
    evaluations: tuple[SourceEvaluation, ...]
