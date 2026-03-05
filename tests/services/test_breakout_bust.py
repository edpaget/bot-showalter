import math
from typing import Any

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.breakout_bust import (
    BreakoutPrediction,
    LabelConfig,
    LabeledSeason,
    OutcomeLabel,
)
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.services.breakout_bust import (
    assemble_labeled_dataset,
    evaluate_classifier,
    find_actionability_threshold,
    generate_labels,
    historical_backtest,
    label_distribution,
)


def _make_adp(
    player_id: int = 1,
    season: int = 2023,
    overall_pick: float = 50.0,
    rank: int = 50,
) -> ADP:
    return ADP(
        player_id=player_id,
        season=season,
        provider="espn",
        overall_pick=overall_pick,
        rank=rank,
        positions="OF",
    )


def _make_valuation(
    player_id: int = 1,
    season: int = 2023,
    rank: int = 20,
    player_type: str = "batter",
    value: float = 10.0,
) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=season,
        system="z",
        version="v1",
        projection_system="actual",
        projection_version="v1",
        player_type=player_type,
        position="OF",
        value=value,
        rank=rank,
        category_scores={},
    )


class FakeDatasetAssembler:
    """Fake assembler that returns configurable rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return DatasetHandle(
            dataset_id=1,
            feature_set_id=1,
            table_name="ds_1",
            row_count=len(self._rows),
            seasons=feature_set.seasons,
        )

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        return DatasetSplits(train=handle, validation=None, holdout=None)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.materialize(feature_set)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        return list(self._rows)


# ---------------------------------------------------------------------------
# generate_labels tests
# ---------------------------------------------------------------------------


class TestGenerateLabelsEmpty:
    def test_empty_adp(self) -> None:
        result = generate_labels([], [_make_valuation()], LabelConfig())
        assert result == []

    def test_empty_valuations(self) -> None:
        result = generate_labels([_make_adp()], [], LabelConfig())
        assert result == []

    def test_both_empty(self) -> None:
        result = generate_labels([], [], LabelConfig())
        assert result == []


class TestGenerateLabelsClassification:
    def test_breakout(self) -> None:
        adp = _make_adp(player_id=1, rank=100, overall_pick=100.0)
        val = _make_valuation(player_id=1, rank=50)
        # rank_delta = 100 - 50 = 50 >= 30 → BREAKOUT
        result = generate_labels([adp], [val], LabelConfig())
        assert len(result) == 1
        assert result[0].label is OutcomeLabel.BREAKOUT
        assert result[0].rank_delta == 50

    def test_bust(self) -> None:
        adp = _make_adp(player_id=1, rank=50, overall_pick=50.0)
        val = _make_valuation(player_id=1, rank=100)
        # rank_delta = 50 - 100 = -50 <= -30 → BUST
        result = generate_labels([adp], [val], LabelConfig())
        assert len(result) == 1
        assert result[0].label is OutcomeLabel.BUST
        assert result[0].rank_delta == -50

    def test_neutral(self) -> None:
        adp = _make_adp(player_id=1, rank=50, overall_pick=50.0)
        val = _make_valuation(player_id=1, rank=40)
        # rank_delta = 50 - 40 = 10 → NEUTRAL
        result = generate_labels([adp], [val], LabelConfig())
        assert len(result) == 1
        assert result[0].label is OutcomeLabel.NEUTRAL
        assert result[0].rank_delta == 10


class TestGenerateLabelsThresholdBoundary:
    def test_exact_breakout_threshold(self) -> None:
        adp = _make_adp(player_id=1, rank=80, overall_pick=80.0)
        val = _make_valuation(player_id=1, rank=50)
        # rank_delta = 80 - 50 = 30 == breakout_threshold → BREAKOUT
        result = generate_labels([adp], [val], LabelConfig())
        assert result[0].label is OutcomeLabel.BREAKOUT

    def test_just_below_breakout_threshold(self) -> None:
        adp = _make_adp(player_id=1, rank=79, overall_pick=79.0)
        val = _make_valuation(player_id=1, rank=50)
        # rank_delta = 79 - 50 = 29 → NEUTRAL
        result = generate_labels([adp], [val], LabelConfig())
        assert result[0].label is OutcomeLabel.NEUTRAL

    def test_exact_bust_threshold(self) -> None:
        adp = _make_adp(player_id=1, rank=50, overall_pick=50.0)
        val = _make_valuation(player_id=1, rank=80)
        # rank_delta = 50 - 80 = -30 == bust_threshold → BUST
        result = generate_labels([adp], [val], LabelConfig())
        assert result[0].label is OutcomeLabel.BUST

    def test_just_above_bust_threshold(self) -> None:
        adp = _make_adp(player_id=1, rank=50, overall_pick=50.0)
        val = _make_valuation(player_id=1, rank=79)
        # rank_delta = 50 - 79 = -29 → NEUTRAL
        result = generate_labels([adp], [val], LabelConfig())
        assert result[0].label is OutcomeLabel.NEUTRAL


class TestGenerateLabelsFiltering:
    def test_min_adp_rank_filters(self) -> None:
        adp = _make_adp(player_id=1, rank=301, overall_pick=301.0)
        val = _make_valuation(player_id=1, rank=100)
        result = generate_labels([adp], [val], LabelConfig(min_adp_rank=300))
        assert result == []

    def test_min_adp_rank_boundary_included(self) -> None:
        adp = _make_adp(player_id=1, rank=300, overall_pick=300.0)
        val = _make_valuation(player_id=1, rank=100)
        result = generate_labels([adp], [val], LabelConfig(min_adp_rank=300))
        assert len(result) == 1

    def test_no_match_excluded(self) -> None:
        adp = _make_adp(player_id=1, rank=50, overall_pick=50.0)
        val = _make_valuation(player_id=2, rank=20)
        result = generate_labels([adp], [val], LabelConfig())
        assert result == []


class TestGenerateLabelsDedup:
    def test_adp_dedup_keeps_lowest_pick(self) -> None:
        adp1 = _make_adp(player_id=1, rank=100, overall_pick=100.0)
        adp2 = _make_adp(player_id=1, rank=80, overall_pick=80.0)
        val = _make_valuation(player_id=1, rank=50)
        result = generate_labels([adp1, adp2], [val], LabelConfig())
        assert len(result) == 1
        assert result[0].adp_rank == 80
        assert result[0].adp_pick == 80.0


class TestGenerateLabelsCustomConfig:
    def test_custom_thresholds(self) -> None:
        config = LabelConfig(breakout_threshold=10, bust_threshold=-10)
        adp = _make_adp(player_id=1, rank=50, overall_pick=50.0)
        val = _make_valuation(player_id=1, rank=40)
        # rank_delta = 50 - 40 = 10 >= 10 → BREAKOUT with custom config
        result = generate_labels([adp], [val], config)
        assert result[0].label is OutcomeLabel.BREAKOUT


class TestGenerateLabelsPlayerType:
    def test_player_type_from_valuation(self) -> None:
        adp = _make_adp(player_id=1, rank=50, overall_pick=50.0)
        val = _make_valuation(player_id=1, rank=20, player_type="pitcher")
        result = generate_labels([adp], [val], LabelConfig())
        assert result[0].player_type == "pitcher"


class TestGenerateLabelsMultiPlayer:
    def test_multiple_players(self) -> None:
        adps = [
            _make_adp(player_id=1, rank=100, overall_pick=100.0),
            _make_adp(player_id=2, rank=50, overall_pick=50.0),
            _make_adp(player_id=3, rank=75, overall_pick=75.0),
        ]
        vals = [
            _make_valuation(player_id=1, rank=50),  # delta=50 → BREAKOUT
            _make_valuation(player_id=2, rank=100),  # delta=-50 → BUST
            _make_valuation(player_id=3, rank=70),  # delta=5 → NEUTRAL
        ]
        result = generate_labels(adps, vals, LabelConfig())
        labels_by_id = {r.player_id: r.label for r in result}
        assert labels_by_id[1] is OutcomeLabel.BREAKOUT
        assert labels_by_id[2] is OutcomeLabel.BUST
        assert labels_by_id[3] is OutcomeLabel.NEUTRAL


# ---------------------------------------------------------------------------
# assemble_labeled_dataset tests
# ---------------------------------------------------------------------------


class TestAssembleLabeledDataset:
    def _make_feature_set(self) -> FeatureSet:
        return FeatureSet(name="test", features=(), seasons=(2023,))

    def test_join_matches(self) -> None:
        labels = [
            LabeledSeason(
                player_id=1,
                season=2023,
                player_type="batter",
                adp_rank=50,
                adp_pick=50.0,
                actual_value_rank=20,
                rank_delta=30,
                label=OutcomeLabel.BREAKOUT,
            ),
        ]
        rows = [{"player_id": 1, "season": 2023, "feature_a": 0.5}]
        assembler = FakeDatasetAssembler(rows)
        result = assemble_labeled_dataset(labels, assembler, self._make_feature_set())
        assert len(result) == 1
        assert result[0]["label"] == "breakout"
        assert result[0]["rank_delta"] == 30
        assert result[0]["adp_rank"] == 50
        assert result[0]["adp_pick"] == 50.0
        assert result[0]["feature_a"] == 0.5

    def test_unmatched_excluded(self) -> None:
        labels = [
            LabeledSeason(
                player_id=1,
                season=2023,
                player_type="batter",
                adp_rank=50,
                adp_pick=50.0,
                actual_value_rank=20,
                rank_delta=30,
                label=OutcomeLabel.BREAKOUT,
            ),
        ]
        rows = [{"player_id=": 2, "season": 2023, "feature_a": 0.5}]
        assembler = FakeDatasetAssembler(rows)
        result = assemble_labeled_dataset(labels, assembler, self._make_feature_set())
        assert result == []

    def test_feature_columns_preserved(self) -> None:
        labels = [
            LabeledSeason(
                player_id=1,
                season=2023,
                player_type="batter",
                adp_rank=50,
                adp_pick=50.0,
                actual_value_rank=20,
                rank_delta=30,
                label=OutcomeLabel.BREAKOUT,
            ),
        ]
        rows = [{"player_id": 1, "season": 2023, "x": 1.0, "y": 2.0}]
        assembler = FakeDatasetAssembler(rows)
        result = assemble_labeled_dataset(labels, assembler, self._make_feature_set())
        assert result[0]["x"] == 1.0
        assert result[0]["y"] == 2.0

    def test_empty_labels(self) -> None:
        rows = [{"player_id": 1, "season": 2023, "x": 1.0}]
        assembler = FakeDatasetAssembler(rows)
        result = assemble_labeled_dataset([], assembler, self._make_feature_set())
        assert result == []

    def test_empty_features(self) -> None:
        labels = [
            LabeledSeason(
                player_id=1,
                season=2023,
                player_type="batter",
                adp_rank=50,
                adp_pick=50.0,
                actual_value_rank=20,
                rank_delta=30,
                label=OutcomeLabel.BREAKOUT,
            ),
        ]
        assembler = FakeDatasetAssembler([])
        result = assemble_labeled_dataset(labels, assembler, self._make_feature_set())
        assert result == []


# ---------------------------------------------------------------------------
# label_distribution tests
# ---------------------------------------------------------------------------


class TestLabelDistribution:
    def test_counts(self) -> None:
        labels = [
            LabeledSeason(1, 2023, "batter", 50, 50.0, 20, 30, OutcomeLabel.BREAKOUT),
            LabeledSeason(2, 2023, "batter", 50, 50.0, 100, -50, OutcomeLabel.BUST),
            LabeledSeason(3, 2023, "batter", 50, 50.0, 45, 5, OutcomeLabel.NEUTRAL),
            LabeledSeason(4, 2023, "batter", 50, 50.0, 40, 10, OutcomeLabel.NEUTRAL),
        ]
        dist = label_distribution(labels)
        assert dist == {"breakout": 1, "bust": 1, "neutral": 2}

    def test_empty(self) -> None:
        dist = label_distribution([])
        assert dist == {"breakout": 0, "bust": 0, "neutral": 0}


# ---------------------------------------------------------------------------
# helpers for evaluation tests
# ---------------------------------------------------------------------------


def _make_label(
    player_id: int,
    label: OutcomeLabel,
    season: int = 2023,
) -> LabeledSeason:
    rank_delta = 50 if label is OutcomeLabel.BREAKOUT else (-50 if label is OutcomeLabel.BUST else 5)
    return LabeledSeason(
        player_id=player_id,
        season=season,
        player_type="batter",
        adp_rank=100,
        adp_pick=100.0,
        actual_value_rank=100 - rank_delta,
        rank_delta=rank_delta,
        label=label,
    )


def _make_prediction(
    player_id: int,
    p_breakout: float = 0.2,
    p_bust: float = 0.2,
    p_neutral: float = 0.6,
) -> BreakoutPrediction:
    return BreakoutPrediction(
        player_id=player_id,
        player_name=f"Player {player_id}",
        player_type="batter",
        position="OF",
        p_breakout=p_breakout,
        p_bust=p_bust,
        p_neutral=p_neutral,
    )


# ---------------------------------------------------------------------------
# evaluate_classifier tests
# ---------------------------------------------------------------------------


class TestEvaluateClassifierThresholds:
    def test_precision_recall_at_threshold(self) -> None:
        """With 10 players: 3 breakout, 7 neutral.
        Predictions: breakouts get p_breakout=0.6, neutrals get 0.1.
        At threshold 0.5: flagged=3, TP=3, precision=1.0, recall=1.0.
        """
        labels = [_make_label(i, OutcomeLabel.BREAKOUT) for i in range(1, 4)]
        labels += [_make_label(i, OutcomeLabel.NEUTRAL) for i in range(4, 11)]

        preds = [_make_prediction(i, p_breakout=0.6, p_bust=0.05, p_neutral=0.35) for i in range(1, 4)]
        preds += [_make_prediction(i, p_breakout=0.1, p_bust=0.1, p_neutral=0.8) for i in range(4, 11)]

        result = evaluate_classifier(labels, preds, thresholds=[0.5])

        breakout_metrics = [m for m in result.threshold_metrics if m.label == "breakout" and m.threshold == 0.5]
        assert len(breakout_metrics) == 1
        m = breakout_metrics[0]
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.flagged == 3
        assert m.true_positives == 3

    def test_imperfect_precision(self) -> None:
        """2 breakouts, 2 neutrals with high p_breakout → precision=0.5."""
        labels = [
            _make_label(1, OutcomeLabel.BREAKOUT),
            _make_label(2, OutcomeLabel.BREAKOUT),
            _make_label(3, OutcomeLabel.NEUTRAL),
            _make_label(4, OutcomeLabel.NEUTRAL),
        ]
        preds = [
            _make_prediction(1, p_breakout=0.7, p_bust=0.1, p_neutral=0.2),
            _make_prediction(2, p_breakout=0.7, p_bust=0.1, p_neutral=0.2),
            _make_prediction(3, p_breakout=0.6, p_bust=0.1, p_neutral=0.3),
            _make_prediction(4, p_breakout=0.6, p_bust=0.1, p_neutral=0.3),
        ]
        result = evaluate_classifier(labels, preds, thresholds=[0.5])
        m = [m for m in result.threshold_metrics if m.label == "breakout"][0]
        assert m.flagged == 4
        assert m.true_positives == 2
        assert m.precision == 0.5
        assert m.recall == 1.0

    def test_bust_threshold(self) -> None:
        """Verify bust thresholds work too."""
        labels = [_make_label(1, OutcomeLabel.BUST), _make_label(2, OutcomeLabel.NEUTRAL)]
        preds = [
            _make_prediction(1, p_breakout=0.1, p_bust=0.7, p_neutral=0.2),
            _make_prediction(2, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
        ]
        result = evaluate_classifier(labels, preds, thresholds=[0.5])
        bust = [m for m in result.threshold_metrics if m.label == "bust"][0]
        assert bust.flagged == 1
        assert bust.true_positives == 1
        assert bust.precision == 1.0
        assert bust.recall == 1.0

    def test_f1_computation(self) -> None:
        """F1 = 2 * prec * recall / (prec + recall)."""
        labels = [
            _make_label(1, OutcomeLabel.BREAKOUT),
            _make_label(2, OutcomeLabel.BREAKOUT),
            _make_label(3, OutcomeLabel.NEUTRAL),
        ]
        preds = [
            _make_prediction(1, p_breakout=0.8, p_bust=0.05, p_neutral=0.15),
            _make_prediction(2, p_breakout=0.2, p_bust=0.3, p_neutral=0.5),  # missed
            _make_prediction(3, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
        ]
        result = evaluate_classifier(labels, preds, thresholds=[0.5])
        m = [m for m in result.threshold_metrics if m.label == "breakout"][0]
        # precision=1/1=1.0, recall=1/2=0.5, F1=2*1.0*0.5/(1.0+0.5)=2/3
        assert m.precision == 1.0
        assert m.recall == 0.5
        assert abs(m.f1 - 2 / 3) < 1e-9


class TestEvaluateClassifierCalibration:
    def test_well_calibrated(self) -> None:
        """All predictions at ~0.3 for breakout, actual rate is 0.3 → bins should align."""
        labels = [_make_label(i, OutcomeLabel.BREAKOUT) for i in range(1, 4)]
        labels += [_make_label(i, OutcomeLabel.NEUTRAL) for i in range(4, 11)]

        # All get p_breakout=0.3
        preds = [_make_prediction(i, p_breakout=0.3, p_bust=0.1, p_neutral=0.6) for i in range(1, 11)]

        result = evaluate_classifier(labels, preds)

        # Find the bin containing 0.3 (center=0.25 or 0.35 depending on binning)
        breakout_bins = [b for b in result.calibration_bins if b.bin_center == 0.25 or b.bin_center == 0.35]
        # At least one bin should have data
        populated = [b for b in breakout_bins if b.count > 0]
        assert len(populated) >= 1
        # The populated bin's mean_actual should be close to 0.3
        for b in populated:
            assert abs(b.mean_actual - 0.3) < 0.01

    def test_bins_cover_range(self) -> None:
        """Calibration bins should have centers from 0.05 to 0.95."""
        labels = [_make_label(1, OutcomeLabel.BREAKOUT), _make_label(2, OutcomeLabel.NEUTRAL)]
        preds = [
            _make_prediction(1, p_breakout=0.8, p_bust=0.1, p_neutral=0.1),
            _make_prediction(2, p_breakout=0.2, p_bust=0.1, p_neutral=0.7),
        ]
        result = evaluate_classifier(labels, preds)
        # Should have bins for breakout and bust
        assert len(result.calibration_bins) > 0


class TestEvaluateClassifierLift:
    def test_positive_lift(self) -> None:
        """Top-2 by p_breakout are both breakouts, base rate is 2/6 → lift > 1."""
        labels = [_make_label(1, OutcomeLabel.BREAKOUT), _make_label(2, OutcomeLabel.BREAKOUT)]
        labels += [_make_label(i, OutcomeLabel.NEUTRAL) for i in range(3, 7)]

        preds = [
            _make_prediction(1, p_breakout=0.9, p_bust=0.05, p_neutral=0.05),
            _make_prediction(2, p_breakout=0.8, p_bust=0.05, p_neutral=0.15),
            _make_prediction(3, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
            _make_prediction(4, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
            _make_prediction(5, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
            _make_prediction(6, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
        ]

        result = evaluate_classifier(labels, preds, top_ns=[2])
        breakout_lifts = [lr for lr in result.lift_results if lr.label == "breakout" and lr.top_n == 2]
        assert len(breakout_lifts) == 1
        lr = breakout_lifts[0]
        assert lr.flagged_rate == 1.0  # 2/2 top-2 are breakout
        assert abs(lr.base_rate - 2 / 6) < 1e-9
        assert lr.lift == 1.0 / (2 / 6)  # 3.0

    def test_no_lift(self) -> None:
        """Top-2 has 0 breakouts → lift = 0."""
        labels = [_make_label(1, OutcomeLabel.BREAKOUT)]
        labels += [_make_label(i, OutcomeLabel.NEUTRAL) for i in range(2, 5)]

        preds = [
            _make_prediction(1, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
            _make_prediction(2, p_breakout=0.9, p_bust=0.05, p_neutral=0.05),
            _make_prediction(3, p_breakout=0.8, p_bust=0.05, p_neutral=0.15),
            _make_prediction(4, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
        ]
        result = evaluate_classifier(labels, preds, top_ns=[2])
        lr = [lr for lr in result.lift_results if lr.label == "breakout" and lr.top_n == 2][0]
        assert lr.flagged_rate == 0.0
        assert lr.lift == 0.0

    def test_top_n_capped_to_population(self) -> None:
        """When top_n > population, uses full population."""
        labels = [_make_label(1, OutcomeLabel.BREAKOUT), _make_label(2, OutcomeLabel.NEUTRAL)]
        preds = [
            _make_prediction(1, p_breakout=0.8, p_bust=0.1, p_neutral=0.1),
            _make_prediction(2, p_breakout=0.2, p_bust=0.1, p_neutral=0.7),
        ]
        result = evaluate_classifier(labels, preds, top_ns=[100])
        lr = [lr for lr in result.lift_results if lr.label == "breakout"][0]
        # top_n capped to 2, rate=1/2=0.5, base_rate=1/2=0.5, lift=1.0
        assert lr.flagged_rate == 0.5
        assert lr.lift == 1.0


class TestEvaluateClassifierLogLoss:
    def test_perfect_predictions(self) -> None:
        """Perfect predictions should have very low log-loss."""
        labels = [
            _make_label(1, OutcomeLabel.BREAKOUT),
            _make_label(2, OutcomeLabel.BUST),
            _make_label(3, OutcomeLabel.NEUTRAL),
        ]
        preds = [
            _make_prediction(1, p_breakout=0.98, p_bust=0.01, p_neutral=0.01),
            _make_prediction(2, p_breakout=0.01, p_bust=0.98, p_neutral=0.01),
            _make_prediction(3, p_breakout=0.01, p_bust=0.01, p_neutral=0.98),
        ]
        result = evaluate_classifier(labels, preds)
        assert result.log_loss < 0.1
        assert result.log_loss < result.base_rate_log_loss

    def test_log_loss_computed(self) -> None:
        """Verify log-loss is a positive number."""
        labels = [_make_label(1, OutcomeLabel.BREAKOUT), _make_label(2, OutcomeLabel.NEUTRAL)]
        preds = [
            _make_prediction(1, p_breakout=0.5, p_bust=0.2, p_neutral=0.3),
            _make_prediction(2, p_breakout=0.3, p_bust=0.3, p_neutral=0.4),
        ]
        result = evaluate_classifier(labels, preds)
        assert result.log_loss > 0
        assert result.base_rate_log_loss > 0
        assert math.isfinite(result.log_loss)

    def test_n_evaluated(self) -> None:
        labels = [_make_label(1, OutcomeLabel.BREAKOUT), _make_label(2, OutcomeLabel.NEUTRAL)]
        preds = [
            _make_prediction(1, p_breakout=0.5, p_bust=0.2, p_neutral=0.3),
            _make_prediction(2, p_breakout=0.3, p_bust=0.3, p_neutral=0.4),
            _make_prediction(99, p_breakout=0.3, p_bust=0.3, p_neutral=0.4),  # no label match
        ]
        result = evaluate_classifier(labels, preds)
        assert result.n_evaluated == 2


class TestEvaluateClassifierEmpty:
    def test_empty_labels(self) -> None:
        result = evaluate_classifier([], [_make_prediction(1)])
        assert result.n_evaluated == 0
        assert result.threshold_metrics == []
        assert result.calibration_bins == []
        assert result.lift_results == []
        assert result.log_loss == 0.0
        assert result.base_rate_log_loss == 0.0

    def test_empty_predictions(self) -> None:
        result = evaluate_classifier([_make_label(1, OutcomeLabel.BREAKOUT)], [])
        assert result.n_evaluated == 0

    def test_no_overlap(self) -> None:
        result = evaluate_classifier(
            [_make_label(1, OutcomeLabel.BREAKOUT)],
            [_make_prediction(99)],
        )
        assert result.n_evaluated == 0


# ---------------------------------------------------------------------------
# historical_backtest tests
# ---------------------------------------------------------------------------


class TestHistoricalBacktest:
    def test_multiple_seasons(self) -> None:
        labels_2022 = [_make_label(1, OutcomeLabel.BREAKOUT, season=2022)]
        labels_2023 = [_make_label(2, OutcomeLabel.NEUTRAL, season=2023)]
        preds_2022 = [_make_prediction(1, p_breakout=0.8, p_bust=0.1, p_neutral=0.1)]
        preds_2023 = [_make_prediction(2, p_breakout=0.2, p_bust=0.1, p_neutral=0.7)]

        results = historical_backtest(
            {2022: labels_2022, 2023: labels_2023},
            {2022: preds_2022, 2023: preds_2023},
        )
        assert 2022 in results
        assert 2023 in results
        assert results[2022].n_evaluated == 1
        assert results[2023].n_evaluated == 1

    def test_missing_season_skipped(self) -> None:
        """Season with labels but no predictions is skipped."""
        labels = {2022: [_make_label(1, OutcomeLabel.BREAKOUT, season=2022)]}
        preds: dict[int, list[BreakoutPrediction]] = {}
        results = historical_backtest(labels, preds)
        assert 2022 not in results


# ---------------------------------------------------------------------------
# find_actionability_threshold tests
# ---------------------------------------------------------------------------


class TestFindActionabilityThreshold:
    def test_finds_lowest_threshold(self) -> None:
        """Should return lowest threshold meeting min_precision."""
        labels = [_make_label(i, OutcomeLabel.BREAKOUT) for i in range(1, 4)]
        labels += [_make_label(i, OutcomeLabel.NEUTRAL) for i in range(4, 11)]
        preds = [_make_prediction(i, p_breakout=0.7, p_bust=0.1, p_neutral=0.2) for i in range(1, 4)]
        preds += [_make_prediction(i, p_breakout=0.1, p_bust=0.1, p_neutral=0.8) for i in range(4, 11)]

        evaluation = evaluate_classifier(labels, preds, thresholds=[0.1, 0.3, 0.5])
        threshold = find_actionability_threshold(evaluation, "breakout", min_precision=0.4)
        assert threshold is not None
        assert threshold <= 0.5  # Should find a working threshold

    def test_no_threshold_meets_criteria(self) -> None:
        """All thresholds have precision < min → returns None."""
        labels = [_make_label(1, OutcomeLabel.BREAKOUT)]
        labels += [_make_label(i, OutcomeLabel.NEUTRAL) for i in range(2, 20)]
        # All get high p_breakout → precision is very low
        preds = [_make_prediction(i, p_breakout=0.6, p_bust=0.1, p_neutral=0.3) for i in range(1, 20)]

        evaluation = evaluate_classifier(labels, preds, thresholds=[0.1, 0.3, 0.5])
        threshold = find_actionability_threshold(evaluation, "breakout", min_precision=0.99)
        assert threshold is None

    def test_label_filter(self) -> None:
        """Should only consider metrics for the specified label."""
        labels = [_make_label(1, OutcomeLabel.BUST), _make_label(2, OutcomeLabel.NEUTRAL)]
        preds = [
            _make_prediction(1, p_breakout=0.1, p_bust=0.8, p_neutral=0.1),
            _make_prediction(2, p_breakout=0.1, p_bust=0.1, p_neutral=0.8),
        ]
        evaluation = evaluate_classifier(labels, preds, thresholds=[0.5])
        threshold = find_actionability_threshold(evaluation, "bust", min_precision=0.4)
        assert threshold is not None
