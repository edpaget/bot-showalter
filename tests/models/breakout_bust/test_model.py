import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest

from fantasy_baseball_manager.domain import LabeledSeason, OutcomeLabel
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.breakout_bust.model import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    BreakoutBustModel,
    _renormalize_probabilities,
)
from fantasy_baseball_manager.models.protocols import Evaluable, Model, ModelConfig
from fantasy_baseball_manager.models.registry import get, register

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = ["age", "pa_1", "hr_1", "avg_exit_velo", "whiff_rate"]

_MODEL_PARAMS: dict[str, Any] = {
    "feature_columns": {
        "batter": FEATURE_COLS,
        "pitcher": FEATURE_COLS,
    },
}


def _make_labeled_season(
    player_id: int,
    season: int,
    player_type: str,
    label: OutcomeLabel,
    adp_rank: int = 50,
) -> LabeledSeason:
    rank_delta = 40 if label is OutcomeLabel.BREAKOUT else (-40 if label is OutcomeLabel.BUST else 0)
    return LabeledSeason(
        player_id=player_id,
        season=season,
        player_type=player_type,
        adp_rank=adp_rank,
        adp_pick=float(adp_rank),
        actual_value_rank=adp_rank - rank_delta,
        rank_delta=rank_delta,
        label=label,
    )


def _make_feature_row(
    player_id: int,
    season: int,
    label: OutcomeLabel,
    rng: random.Random,
) -> dict[str, Any]:
    """Create a feature row with biased values based on label.

    Breakout players get +2.0 bias, bust players get -2.0 bias,
    so the classifier has strong learnable signal.
    """
    bias = 2.0 if label is OutcomeLabel.BREAKOUT else (-2.0 if label is OutcomeLabel.BUST else 0.0)
    return {
        "player_id": player_id,
        "season": season,
        "age": 25.0 + rng.gauss(0, 1) + bias,
        "pa_1": 500.0 + rng.gauss(0, 20) + bias * 20,
        "hr_1": 20.0 + rng.gauss(0, 2) + bias * 2,
        "avg_exit_velo": 88.0 + rng.gauss(0, 1) + bias,
        "whiff_rate": 0.25 + rng.gauss(0, 0.02) - bias * 0.02,
        "adp_rank": 50,
        "adp_pick": 50.0,
    }


def _generate_synthetic_data(
    seasons: list[int],
    players_per_season: int = 60,
    player_type: str = "batter",
) -> tuple[list[LabeledSeason], list[dict[str, Any]]]:
    """Generate synthetic labels and feature rows for testing."""
    rng = random.Random(42)  # noqa: S311
    labels: list[LabeledSeason] = []
    rows: list[dict[str, Any]] = []

    label_choices = [OutcomeLabel.BREAKOUT] * 10 + [OutcomeLabel.BUST] * 10 + [OutcomeLabel.NEUTRAL] * 40

    for season in seasons:
        for i in range(players_per_season):
            pid = season * 1000 + i
            label = rng.choice(label_choices)
            labels.append(_make_labeled_season(pid, season, player_type, label, adp_rank=i + 1))
            rows.append(_make_feature_row(pid, season, label, rng))

    return labels, rows


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLabelSource:
    """Returns configurable labels per season."""

    def __init__(self, labels: list[LabeledSeason]) -> None:
        self._by_season: dict[int, list[LabeledSeason]] = {}
        for ls in labels:
            self._by_season.setdefault(ls.season, []).append(ls)

    def get_labels(self, season: int) -> list[LabeledSeason]:
        return self._by_season.get(season, [])


class FakeAssembler:
    """Returns configurable rows regardless of feature set."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self._counter = 0

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        self._counter += 1
        return DatasetHandle(
            dataset_id=self._counter,
            feature_set_id=self._counter,
            table_name="fake",
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
        raise NotImplementedError

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.materialize(feature_set)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        return self._rows


# ---------------------------------------------------------------------------
# Label encoding tests
# ---------------------------------------------------------------------------


class TestLabelEncoding:
    def test_label_to_int_mapping(self) -> None:
        assert LABEL_TO_INT[OutcomeLabel.NEUTRAL] == 0
        assert LABEL_TO_INT[OutcomeLabel.BREAKOUT] == 1
        assert LABEL_TO_INT[OutcomeLabel.BUST] == 2

    def test_int_to_label_mapping(self) -> None:
        assert INT_TO_LABEL[0] is OutcomeLabel.NEUTRAL
        assert INT_TO_LABEL[1] is OutcomeLabel.BREAKOUT
        assert INT_TO_LABEL[2] is OutcomeLabel.BUST

    def test_round_trip(self) -> None:
        for label in OutcomeLabel:
            assert INT_TO_LABEL[LABEL_TO_INT[label]] is label


# ---------------------------------------------------------------------------
# Protocol and registry tests
# ---------------------------------------------------------------------------


class TestBreakoutBustModelProtocol:
    def test_satisfies_model_protocol(self) -> None:
        model = BreakoutBustModel.__new__(BreakoutBustModel)
        assert isinstance(model, Model)

    def test_properties(self) -> None:
        model = BreakoutBustModel.__new__(BreakoutBustModel)
        assert model.name == "breakout-bust"
        assert model.description
        assert "train" in model.supported_operations
        assert "predict" in model.supported_operations
        assert model.artifact_type == "breakout-bust-classifier"


class TestBreakoutBustModelRegistered:
    def test_registered_in_registry(self, isolated_model_registry: None) -> None:
        register("breakout-bust")(BreakoutBustModel)
        cls = get("breakout-bust")
        assert cls is BreakoutBustModel


# ---------------------------------------------------------------------------
# Train tests
# ---------------------------------------------------------------------------


def _make_model_and_config(
    tmp_path: Path,
    all_labels: list[LabeledSeason],
    all_rows: list[dict[str, Any]],
    seasons: list[int],
) -> tuple[BreakoutBustModel, ModelConfig]:
    model = BreakoutBustModel(
        assembler=FakeAssembler(all_rows),
        label_source=FakeLabelSource(all_labels),
    )
    config = ModelConfig(
        artifacts_dir=str(tmp_path),
        seasons=seasons,
        model_params=_MODEL_PARAMS,
    )
    return model, config


class TestBreakoutBustModelTrain:
    @pytest.fixture
    def _combined_data(self) -> tuple[list[LabeledSeason], list[dict[str, Any]]]:
        batter_labels, batter_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023],
            players_per_season=60,
            player_type="batter",
        )
        pitcher_labels, pitcher_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023],
            players_per_season=60,
            player_type="pitcher",
        )
        return batter_labels + pitcher_labels, batter_rows + pitcher_rows

    def test_train_requires_at_least_2_seasons(self, tmp_path: Path) -> None:
        labels, rows = _generate_synthetic_data([2023], players_per_season=10, player_type="batter")
        model, config = _make_model_and_config(tmp_path, labels, rows, seasons=[2023])
        with pytest.raises(ValueError, match="at least 2"):
            model.train(config)

    def test_train_produces_metrics(
        self,
        tmp_path: Path,
        _combined_data: tuple[list[LabeledSeason], list[dict[str, Any]]],
    ) -> None:
        all_labels, all_rows = _combined_data
        model, config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        result = model.train(config)

        assert result.model_name == "breakout-bust"
        assert "batter_log_loss" in result.metrics
        assert "pitcher_log_loss" in result.metrics
        assert "batter_base_rate_log_loss" in result.metrics
        assert "pitcher_base_rate_log_loss" in result.metrics

    def test_train_log_loss_beats_base_rate(
        self,
        tmp_path: Path,
        _combined_data: tuple[list[LabeledSeason], list[dict[str, Any]]],
    ) -> None:
        all_labels, all_rows = _combined_data
        model, config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        result = model.train(config)

        for ptype in ("batter", "pitcher"):
            assert result.metrics[f"{ptype}_log_loss"] <= result.metrics[f"{ptype}_base_rate_log_loss"]

    def test_train_saves_artifacts(
        self,
        tmp_path: Path,
        _combined_data: tuple[list[LabeledSeason], list[dict[str, Any]]],
    ) -> None:
        all_labels, all_rows = _combined_data
        model, config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        model.train(config)

        artifact_path = tmp_path / "breakout-bust-classifier"
        assert (artifact_path / "batter_classifier.joblib").exists()
        assert (artifact_path / "pitcher_classifier.joblib").exists()

    def test_train_cv_metrics_with_3_plus_seasons(
        self,
        tmp_path: Path,
        _combined_data: tuple[list[LabeledSeason], list[dict[str, Any]]],
    ) -> None:
        all_labels, all_rows = _combined_data
        model, config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        result = model.train(config)

        assert "batter_cv_log_loss" in result.metrics
        assert "pitcher_cv_log_loss" in result.metrics

    def test_train_saves_training_metadata(
        self,
        tmp_path: Path,
        _combined_data: tuple[list[LabeledSeason], list[dict[str, Any]]],
    ) -> None:
        all_labels, all_rows = _combined_data
        model, config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        model.train(config)
        assert (tmp_path / "breakout-bust-classifier" / "training_metadata.json").exists()

    def test_predict_raises_on_leakage(
        self,
        tmp_path: Path,
        _combined_data: tuple[list[LabeledSeason], list[dict[str, Any]]],
    ) -> None:
        all_labels, all_rows = _combined_data
        model, config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        model.train(config)
        predict_config = ModelConfig(artifacts_dir=str(tmp_path), seasons=[2023], model_params=_MODEL_PARAMS)
        with pytest.raises(ValueError, match="Data leakage"):
            model.predict(predict_config)


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


class TestBreakoutBustModelPredict:
    @pytest.fixture
    def _trained_model(self, tmp_path: Path) -> BreakoutBustModel:
        """Train a model and return it."""
        batter_labels, batter_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023, 2024],
            players_per_season=60,
            player_type="batter",
        )
        pitcher_labels, pitcher_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023, 2024],
            players_per_season=60,
            player_type="pitcher",
        )
        all_labels = batter_labels + pitcher_labels
        all_rows = batter_rows + pitcher_rows

        model, train_config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        model.train(train_config)
        return model

    def _predict_config(self, tmp_path: Path) -> ModelConfig:
        return ModelConfig(
            artifacts_dir=str(tmp_path),
            seasons=[2024],
            model_params=_MODEL_PARAMS,
        )

    def test_predict_returns_predictions(
        self,
        tmp_path: Path,
        _trained_model: BreakoutBustModel,
    ) -> None:
        result = _trained_model.predict(self._predict_config(tmp_path))
        assert result.model_name == "breakout-bust"
        assert len(result.predictions) > 0

    def test_predict_probabilities_sum_to_one(
        self,
        tmp_path: Path,
        _trained_model: BreakoutBustModel,
    ) -> None:
        result = _trained_model.predict(self._predict_config(tmp_path))
        for pred in result.predictions:
            total = pred["p_breakout"] + pred["p_bust"] + pred["p_neutral"]
            assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, expected ~1.0"

    def test_predict_top_features_populated(
        self,
        tmp_path: Path,
        _trained_model: BreakoutBustModel,
    ) -> None:
        result = _trained_model.predict(self._predict_config(tmp_path))
        for pred in result.predictions:
            assert len(pred["top_features"]) > 0
            # Verify sorted descending by importance
            importances = [imp for _, imp in pred["top_features"]]
            assert importances == sorted(importances, reverse=True)

    def test_predict_both_player_types(
        self,
        tmp_path: Path,
        _trained_model: BreakoutBustModel,
    ) -> None:
        result = _trained_model.predict(self._predict_config(tmp_path))
        player_types = {p["player_type"] for p in result.predictions}
        assert "batter" in player_types
        assert "pitcher" in player_types


# ---------------------------------------------------------------------------
# Evaluate tests
# ---------------------------------------------------------------------------


class TestBreakoutBustModelEvaluate:
    def test_supports_evaluate_operation(self) -> None:
        model = BreakoutBustModel.__new__(BreakoutBustModel)
        assert "evaluate" in model.supported_operations

    def test_satisfies_evaluable_protocol(self) -> None:
        model = BreakoutBustModel.__new__(BreakoutBustModel)
        assert isinstance(model, Evaluable)

    def test_evaluate_produces_system_metrics(self, tmp_path: Path) -> None:
        batter_labels, batter_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023],
            players_per_season=60,
            player_type="batter",
        )
        pitcher_labels, pitcher_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023],
            players_per_season=60,
            player_type="pitcher",
        )
        all_labels = batter_labels + pitcher_labels
        all_rows = batter_rows + pitcher_rows

        model, config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        result = model.evaluate(config)

        assert result.system == "breakout-bust"
        assert result.version == ""
        assert "log_loss" in result.metrics
        assert "base_rate_log_loss" in result.metrics

    def test_evaluate_requires_at_least_2_seasons(self, tmp_path: Path) -> None:
        labels, rows = _generate_synthetic_data([2023], players_per_season=10, player_type="batter")
        model, config = _make_model_and_config(tmp_path, labels, rows, seasons=[2023])
        with pytest.raises(ValueError, match="at least 2"):
            model.evaluate(config)


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestRenormalizeProbabilities:
    def test_already_normalized(self) -> None:
        p_bo, p_bu, p_ne = _renormalize_probabilities(
            np.array([0.2, 0.3]),
            np.array([0.1, 0.2]),
        )
        np.testing.assert_allclose(p_bo + p_bu + p_ne, 1.0)

    def test_clamps_negatives(self) -> None:
        p_bo, p_bu, p_ne = _renormalize_probabilities(
            np.array([-0.1, 0.5]),
            np.array([0.3, -0.2]),
        )
        assert np.all(p_bo >= 0)
        assert np.all(p_bu >= 0)
        assert np.all(p_ne >= 0)
        np.testing.assert_allclose(p_bo + p_bu + p_ne, 1.0)

    def test_renormalizes_when_sum_exceeds_one(self) -> None:
        p_bo, p_bu, p_ne = _renormalize_probabilities(
            np.array([0.6]),
            np.array([0.7]),
        )
        np.testing.assert_allclose(p_bo + p_bu + p_ne, 1.0)
        # p_neutral should be 0 since 0.6 + 0.7 > 1
        np.testing.assert_allclose(p_ne, 0.0, atol=1e-10)


class TestCalibration:
    @pytest.fixture
    def _combined_data(self) -> tuple[list[LabeledSeason], list[dict[str, Any]]]:
        batter_labels, batter_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023],
            players_per_season=60,
            player_type="batter",
        )
        pitcher_labels, pitcher_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023],
            players_per_season=60,
            player_type="pitcher",
        )
        return batter_labels + pitcher_labels, batter_rows + pitcher_rows

    def test_train_saves_calibrators(
        self,
        tmp_path: Path,
        _combined_data: tuple[list[LabeledSeason], list[dict[str, Any]]],
    ) -> None:
        all_labels, all_rows = _combined_data
        model, config = _make_model_and_config(tmp_path, all_labels, all_rows, seasons=[2020, 2021, 2022, 2023])
        model.train(config)

        artifact_path = tmp_path / "breakout-bust-classifier"
        assert (artifact_path / "batter_calibrators.joblib").exists()
        assert (artifact_path / "pitcher_calibrators.joblib").exists()

    def test_predict_applies_calibration(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify calibrated predictions differ from raw and still sum to ~1."""
        batter_labels, batter_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023, 2024],
            players_per_season=60,
            player_type="batter",
        )
        pitcher_labels, pitcher_rows = _generate_synthetic_data(
            seasons=[2020, 2021, 2022, 2023, 2024],
            players_per_season=60,
            player_type="pitcher",
        )
        all_labels = batter_labels + pitcher_labels
        all_rows = batter_rows + pitcher_rows

        model, train_config = _make_model_and_config(
            tmp_path,
            all_labels,
            all_rows,
            seasons=[2020, 2021, 2022, 2023],
        )
        model.train(train_config)

        predict_config = ModelConfig(
            artifacts_dir=str(tmp_path),
            seasons=[2024],
            model_params=_MODEL_PARAMS,
        )
        result = model.predict(predict_config)

        for pred in result.predictions:
            total = pred["p_breakout"] + pred["p_bust"] + pred["p_neutral"]
            assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, expected ~1.0"

    def test_predict_graceful_without_calibrator_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Predict works if no calibrator files exist (e.g. trained with only 2 seasons)."""
        batter_labels, batter_rows = _generate_synthetic_data(
            seasons=[2022, 2023, 2024],
            players_per_season=60,
            player_type="batter",
        )
        pitcher_labels, pitcher_rows = _generate_synthetic_data(
            seasons=[2022, 2023, 2024],
            players_per_season=60,
            player_type="pitcher",
        )
        all_labels = batter_labels + pitcher_labels
        all_rows = batter_rows + pitcher_rows

        # Train with only 2 seasons — no calibrators saved
        model, train_config = _make_model_and_config(
            tmp_path,
            all_labels,
            all_rows,
            seasons=[2022, 2023],
        )
        model.train(train_config)

        # Remove calibrator files if they exist (shouldn't with 2 seasons, but be safe)
        artifact_path = tmp_path / "breakout-bust-classifier"
        for f in artifact_path.glob("*_calibrators.joblib"):
            f.unlink()

        predict_config = ModelConfig(
            artifacts_dir=str(tmp_path),
            seasons=[2024],
            model_params=_MODEL_PARAMS,
        )
        result = model.predict(predict_config)
        assert len(result.predictions) > 0
        for pred in result.predictions:
            total = pred["p_breakout"] + pred["p_bust"] + pred["p_neutral"]
            assert abs(total - 1.0) < 1e-6
