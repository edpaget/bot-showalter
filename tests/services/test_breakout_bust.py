from typing import Any

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.breakout_bust import (
    LabelConfig,
    LabeledSeason,
    OutcomeLabel,
)
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.services.breakout_bust import (
    assemble_labeled_dataset,
    generate_labels,
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
