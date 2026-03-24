import pytest

from fantasy_baseball_manager.domain.breakout_bust import (
    BreakoutPrediction,
    ClassifierCalibrationBin,
    ClassifierEvaluation,
    LabelConfig,
    LabeledSeason,
    LiftResult,
    OutcomeLabel,
    ThresholdMetrics,
)
from fantasy_baseball_manager.domain.identity import PlayerType


class TestOutcomeLabel:
    def test_values(self) -> None:
        assert OutcomeLabel.BREAKOUT == "breakout"
        assert OutcomeLabel.BUST == "bust"
        assert OutcomeLabel.NEUTRAL == "neutral"

    def test_is_str(self) -> None:
        assert isinstance(OutcomeLabel.BREAKOUT, str)

    def test_from_string(self) -> None:
        assert OutcomeLabel("breakout") is OutcomeLabel.BREAKOUT


class TestLabelConfig:
    def test_defaults(self) -> None:
        config = LabelConfig()
        assert config.breakout_threshold == 30
        assert config.bust_threshold == -30
        assert config.min_adp_rank == 300

    def test_custom_values(self) -> None:
        config = LabelConfig(breakout_threshold=50, bust_threshold=-50, min_adp_rank=200)
        assert config.breakout_threshold == 50
        assert config.bust_threshold == -50
        assert config.min_adp_rank == 200

    def test_frozen(self) -> None:
        config = LabelConfig()
        with pytest.raises(AttributeError):
            config.breakout_threshold = 99  # type: ignore[misc]


class TestLabeledSeason:
    def test_fields(self) -> None:
        ls = LabeledSeason(
            player_id=1,
            season=2023,
            player_type=PlayerType.BATTER,
            adp_rank=50,
            adp_pick=50.0,
            actual_value_rank=20,
            rank_delta=30,
            label=OutcomeLabel.BREAKOUT,
        )
        assert ls.player_id == 1
        assert ls.season == 2023
        assert ls.player_type == "batter"
        assert ls.adp_rank == 50
        assert ls.adp_pick == 50.0
        assert ls.actual_value_rank == 20
        assert ls.rank_delta == 30
        assert ls.label is OutcomeLabel.BREAKOUT

    def test_frozen(self) -> None:
        ls = LabeledSeason(
            player_id=1,
            season=2023,
            player_type=PlayerType.BATTER,
            adp_rank=50,
            adp_pick=50.0,
            actual_value_rank=20,
            rank_delta=30,
            label=OutcomeLabel.BREAKOUT,
        )
        with pytest.raises(AttributeError):
            ls.label = OutcomeLabel.BUST  # type: ignore[misc]


class TestBreakoutPrediction:
    def test_fields(self) -> None:
        pred = BreakoutPrediction(
            player_id=42,
            player_name="Mike Trout",
            player_type=PlayerType.BATTER,
            position="OF",
            p_breakout=0.6,
            p_bust=0.1,
            p_neutral=0.3,
            top_features=[("age", 0.25), ("avg_exit_velo", 0.15)],
        )
        assert pred.player_id == 42
        assert pred.player_name == "Mike Trout"
        assert pred.player_type == "batter"
        assert pred.position == "OF"
        assert pred.p_breakout == 0.6
        assert pred.p_bust == 0.1
        assert pred.p_neutral == 0.3
        assert pred.top_features == [("age", 0.25), ("avg_exit_velo", 0.15)]

    def test_frozen(self) -> None:
        pred = BreakoutPrediction(
            player_id=42,
            player_name="Mike Trout",
            player_type=PlayerType.BATTER,
            position="OF",
            p_breakout=0.6,
            p_bust=0.1,
            p_neutral=0.3,
            top_features=[],
        )
        with pytest.raises(AttributeError):
            pred.p_breakout = 0.9  # type: ignore[misc]


class TestThresholdMetrics:
    def test_fields(self) -> None:
        tm = ThresholdMetrics(
            label="breakout",
            threshold=0.3,
            precision=0.5,
            recall=0.8,
            f1=0.615,
            flagged=20,
            true_positives=10,
        )
        assert tm.label == "breakout"
        assert tm.threshold == 0.3
        assert tm.precision == 0.5
        assert tm.recall == 0.8
        assert tm.f1 == 0.615
        assert tm.flagged == 20
        assert tm.true_positives == 10

    def test_frozen(self) -> None:
        tm = ThresholdMetrics("breakout", 0.3, 0.5, 0.8, 0.615, 20, 10)
        with pytest.raises(AttributeError):
            tm.precision = 0.9  # type: ignore[misc]


class TestClassifierCalibrationBin:
    def test_fields(self) -> None:
        cb = ClassifierCalibrationBin(
            bin_center=0.25,
            mean_predicted=0.23,
            mean_actual=0.20,
            count=50,
        )
        assert cb.bin_center == 0.25
        assert cb.mean_predicted == 0.23
        assert cb.mean_actual == 0.20
        assert cb.count == 50

    def test_frozen(self) -> None:
        cb = ClassifierCalibrationBin(0.25, 0.23, 0.20, 50)
        with pytest.raises(AttributeError):
            cb.count = 100  # type: ignore[misc]


class TestLiftResult:
    def test_fields(self) -> None:
        lr = LiftResult(
            label="breakout",
            top_n=20,
            flagged_rate=0.35,
            base_rate=0.18,
            lift=1.944,
        )
        assert lr.label == "breakout"
        assert lr.top_n == 20
        assert lr.flagged_rate == 0.35
        assert lr.base_rate == 0.18
        assert lr.lift == 1.944

    def test_frozen(self) -> None:
        lr = LiftResult("breakout", 20, 0.35, 0.18, 1.944)
        with pytest.raises(AttributeError):
            lr.lift = 2.0  # type: ignore[misc]


class TestClassifierEvaluation:
    def test_fields(self) -> None:
        ev = ClassifierEvaluation(
            threshold_metrics=[],
            calibration_bins=[],
            lift_results=[],
            log_loss=0.65,
            base_rate_log_loss=0.90,
            n_evaluated=100,
        )
        assert ev.threshold_metrics == []
        assert ev.calibration_bins == []
        assert ev.lift_results == []
        assert ev.log_loss == 0.65
        assert ev.base_rate_log_loss == 0.90
        assert ev.n_evaluated == 100

    def test_frozen(self) -> None:
        ev = ClassifierEvaluation([], [], [], 0.65, 0.90, 100)
        with pytest.raises(AttributeError):
            ev.log_loss = 0.5  # type: ignore[misc]
