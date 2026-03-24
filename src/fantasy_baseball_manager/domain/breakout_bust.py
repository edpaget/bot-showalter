from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


class OutcomeLabel(StrEnum):
    BREAKOUT = "breakout"
    BUST = "bust"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class LabelConfig:
    breakout_threshold: int = 30
    bust_threshold: int = -30
    min_adp_rank: int = 300


@dataclass(frozen=True)
class LabeledSeason:
    player_id: int
    season: int
    player_type: PlayerType
    adp_rank: int
    adp_pick: float
    actual_value_rank: int
    rank_delta: int
    label: OutcomeLabel


@dataclass(frozen=True)
class BreakoutPrediction:
    player_id: int
    player_name: str
    player_type: PlayerType
    position: str
    p_breakout: float
    p_bust: float
    p_neutral: float
    top_features: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ThresholdMetrics:
    """Precision/recall/F1 at a single probability threshold for one class."""

    label: str
    threshold: float
    precision: float
    recall: float
    f1: float
    flagged: int
    true_positives: int


@dataclass(frozen=True)
class ClassifierCalibrationBin:
    """One bin of a reliability diagram for the breakout/bust classifier."""

    bin_center: float
    mean_predicted: float
    mean_actual: float
    count: int


@dataclass(frozen=True)
class LiftResult:
    """Lift of top-N flagged candidates vs base rate."""

    label: str
    top_n: int
    flagged_rate: float
    base_rate: float
    lift: float


@dataclass(frozen=True)
class ClassifierEvaluation:
    """Aggregate evaluation of the breakout/bust classifier."""

    threshold_metrics: list[ThresholdMetrics]
    calibration_bins: list[ClassifierCalibrationBin]
    lift_results: list[LiftResult]
    log_loss: float
    base_rate_log_loss: float
    n_evaluated: int
