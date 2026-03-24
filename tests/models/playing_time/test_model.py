from typing import TYPE_CHECKING, Any

import pytest

from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.playing_time.aging import AgingCurve
from fantasy_baseball_manager.models.playing_time.engine import (
    PlayingTimeCoefficients,
    ResidualBuckets,
    ResidualPercentiles,
)
from fantasy_baseball_manager.models.playing_time.ip_calibration import IPCalibrator
from fantasy_baseball_manager.models.playing_time.model import PlayingTimeModel
from fantasy_baseball_manager.models.playing_time.serialization import (
    load_coefficients,
    load_ip_calibrator,
    save_aging_curves,
    save_coefficients,
    save_ip_calibrator,
    save_residual_buckets,
)
from fantasy_baseball_manager.models.protocols import (
    Ablatable,
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Preparable,
    Trainable,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


class FakeAssembler:
    """In-memory assembler for testing."""

    def __init__(self, batting_rows: list[dict[str, Any]], pitching_rows: list[dict[str, Any]] | None = None) -> None:
        self._batting_rows = batting_rows
        self._pitching_rows = pitching_rows or []
        self._next_id = 1

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.get_or_materialize(feature_set)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        rows = self._pitching_rows if "pitching" in feature_set.name else self._batting_rows
        handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=self._next_id,
            table_name=f"ds_{feature_set.name}",
            row_count=len(rows),
            seasons=feature_set.seasons,
        )
        self._next_id += 1
        return handle

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        return DatasetSplits(train=handle, validation=None, holdout=None)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        if "pitching" in handle.table_name:
            return self._pitching_rows
        return self._batting_rows


def _make_batting_row(player_id: int, season: int, pa_1: float = 500.0) -> dict[str, Any]:
    """Create a batting row with all required feature columns."""
    return {
        "player_id": player_id,
        "season": season,
        "age": 28,
        "pa_1": pa_1,
        "pa_2": 450.0,
        "pa_3": 400.0,
        "war_1": 3.0,
        "war_2": 2.5,
        "il_days_1": 0.0,
        "il_days_2": 10.0,
        "il_days_3": 0.0,
        "il_stints_1": 0.0,
        "il_stints_2": 1.0,
        "il_days_3yr": 10.0,
        "il_recurrence": 1.0,
        "pt_trend": 50.0,
        # Phase 1 derived features:
        "war_above_2": 1.0,
        "war_above_4": 0.0,
        "war_below_0": 0.0,
        "il_minor": 0.0,
        "il_moderate": 0.0,
        "il_severe": 0.0,
        "war_trend": 0.5,
        "age_il_interact": 0.0,
        # Consensus playing time:
        "steamer_pa": 620.0,
        "zips_pa": 580.0,
        "consensus_pa": 600.0,
        # Training target:
        "target_pa": pa_1 + 20.0,
    }


def _make_pitching_row(player_id: int, season: int, ip_1: float = 180.0) -> dict[str, Any]:
    """Create a pitching row with all required feature columns."""
    return {
        "player_id": player_id,
        "season": season,
        "age": 27,
        "ip_1": ip_1,
        "ip_2": 170.0,
        "ip_3": 160.0,
        "g_1": 30.0,
        "g_2": 28.0,
        "g_3": 25.0,
        "gs_1": 30.0,
        "war_1": 4.0,
        "war_2": 3.5,
        "il_days_1": 0.0,
        "il_days_2": 0.0,
        "il_days_3": 15.0,
        "il_stints_1": 0.0,
        "il_stints_2": 0.0,
        "il_days_3yr": 15.0,
        "il_recurrence": 0.0,
        "pt_trend": 10.0,
        # Phase 1 derived features:
        "war_above_2": 1.0,
        "war_above_4": 0.0,
        "war_below_0": 0.0,
        "il_minor": 0.0,
        "il_moderate": 0.0,
        "il_severe": 0.0,
        "war_trend": 0.5,
        "age_il_interact": 0.0,
        "starter_ratio": 1.0,
        # Consensus playing time:
        "steamer_ip": 200.0,
        "zips_ip": 190.0,
        "consensus_ip": 195.0,
        # Training target:
        "target_ip": ip_1 + 5.0,
    }


_NULL_ASSEMBLER = FakeAssembler(batting_rows=[], pitching_rows=[])


class TestPlayingTimeModelProtocol:
    def test_is_model(self) -> None:
        assert isinstance(PlayingTimeModel(assembler=_NULL_ASSEMBLER), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(PlayingTimeModel(assembler=_NULL_ASSEMBLER), Preparable)

    def test_is_predictable(self) -> None:
        assert isinstance(PlayingTimeModel(assembler=_NULL_ASSEMBLER), Predictable)

    def test_is_trainable(self) -> None:
        assert isinstance(PlayingTimeModel(assembler=_NULL_ASSEMBLER), Trainable)

    def test_is_not_evaluable(self) -> None:
        assert not isinstance(PlayingTimeModel(assembler=_NULL_ASSEMBLER), Evaluable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(PlayingTimeModel(assembler=_NULL_ASSEMBLER), FineTunable)

    def test_name(self) -> None:
        assert PlayingTimeModel(assembler=_NULL_ASSEMBLER).name == "playing_time"

    def test_supported_operations_includes_train(self) -> None:
        ops = PlayingTimeModel(assembler=_NULL_ASSEMBLER).supported_operations
        assert ops == frozenset({"prepare", "train", "predict", "ablate"})

    def test_artifact_type_is_file(self) -> None:
        assert PlayingTimeModel(assembler=_NULL_ASSEMBLER).artifact_type == "file"


def _train_config(tmp_path: Path) -> ModelConfig:
    """Create a training config with aging_min_samples=1 for small test data."""
    return ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path), model_params={"aging_min_samples": 1})


class TestPlayingTimeTrain:
    def test_train_produces_coefficients_file(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        model.train(_train_config(tmp_path))
        artifact = tmp_path / "playing_time" / "latest" / "pt_coefficients.joblib"
        assert artifact.exists()

    def test_train_returns_r_squared_metrics(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        result = model.train(_train_config(tmp_path))
        assert "r_squared_batter" in result.metrics
        assert "r_squared_pitcher" in result.metrics
        assert result.metrics["r_squared_batter"] >= 0.0
        assert result.metrics["r_squared_pitcher"] >= 0.0

    def test_train_saves_aging_curves_file(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        model.train(_train_config(tmp_path))
        artifact = tmp_path / "playing_time" / "latest" / "pt_aging_curves.joblib"
        assert artifact.exists()

    def test_train_coefficients_include_age_pt_factor(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        model.train(_train_config(tmp_path))
        coefficients = load_coefficients(tmp_path / "playing_time" / "latest" / "pt_coefficients.joblib")
        assert "age_pt_factor" in coefficients["batter"].feature_names
        assert "age_pt_factor" in coefficients["pitcher"].feature_names

    def test_train_metrics_include_peak_age(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        result = model.train(_train_config(tmp_path))
        assert "bat_peak_age" in result.metrics
        assert "pitch_peak_age" in result.metrics

    def test_train_saves_residual_buckets_file(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(30)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(30)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        model.train(_train_config(tmp_path))
        artifact = tmp_path / "playing_time" / "latest" / "pt_residual_buckets.joblib"
        assert artifact.exists()

    def test_train_metrics_include_alpha(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        result = model.train(_train_config(tmp_path))
        assert "alpha_batter" in result.metrics
        assert "alpha_pitcher" in result.metrics

    def test_train_alpha_override(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "alpha": 5.0},
        )
        result = model.train(config)
        assert result.metrics["alpha_batter"] == 5.0
        assert result.metrics["alpha_pitcher"] == 5.0

    def test_train_coefficients_have_alpha(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "alpha": 3.0},
        )
        model.train(config)
        coefficients = load_coefficients(tmp_path / "playing_time" / "latest" / "pt_coefficients.joblib")
        assert coefficients["batter"].alpha == 3.0
        assert coefficients["pitcher"].alpha == 3.0

    def test_train_saves_training_metadata(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        model.train(_train_config(tmp_path))
        metadata_path = tmp_path / "playing_time" / "latest" / "training_metadata.json"
        assert metadata_path.exists()

    def test_predict_raises_on_leakage(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        model.train(_train_config(tmp_path))
        # Playing-time predicts max(seasons)+1, so seasons=[2023] predicts 2024.
        # Training on [2023] means predicting 2023 (seasons=[2022]) would be leakage.
        predict_config = ModelConfig(seasons=[2022], artifacts_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Data leakage"):
            model.predict(predict_config)


def _save_test_coefficients(tmp_path: Path) -> None:
    """Save known coefficients and aging curves for predict tests."""
    batter = PlayingTimeCoefficients(
        feature_names=(
            "age",
            "pa_1",
            "pa_2",
            "pa_3",
            "war_1",
            "war_2",
            "il_days_1",
            "il_days_2",
            "il_days_3",
            "il_stints_1",
            "il_stints_2",
            "il_days_3yr",
            "il_recurrence",
            "pt_trend",
            "age_pt_factor",
            "consensus_pa",
        ),
        coefficients=(0.0, 0.8, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
        intercept=50.0,
        r_squared=0.9,
        player_type=PlayerType.BATTER,
    )
    pitcher = PlayingTimeCoefficients(
        feature_names=(
            "age",
            "ip_1",
            "ip_2",
            "ip_3",
            "g_1",
            "g_2",
            "g_3",
            "gs_1",
            "war_1",
            "war_2",
            "il_days_1",
            "il_days_2",
            "il_days_3",
            "il_stints_1",
            "il_stints_2",
            "il_days_3yr",
            "il_recurrence",
            "pt_trend",
            "age_pt_factor",
            "consensus_ip",
        ),
        coefficients=(
            0.0,
            0.7,
            0.15,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        intercept=20.0,
        r_squared=0.85,
        player_type=PlayerType.PITCHER,
    )
    batter_curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type=PlayerType.BATTER)
    pitcher_curve = AgingCurve(
        peak_age=26.0, improvement_rate=0.008, decline_rate=0.007, player_type=PlayerType.PITCHER
    )
    artifact_dir = tmp_path / "playing_time" / "latest"
    artifact_dir.mkdir(parents=True)
    save_coefficients({"batter": batter, "pitcher": pitcher}, artifact_dir / "pt_coefficients.joblib")
    save_aging_curves({"batter": batter_curve, "pitcher": pitcher_curve}, artifact_dir / "pt_aging_curves.joblib")
    bat_percs = ResidualPercentiles(
        p10=-40.0,
        p25=-15.0,
        p50=0.0,
        p75=20.0,
        p90=50.0,
        count=100,
        std=30.0,
        mean_offset=2.0,
    )
    pitch_percs = ResidualPercentiles(
        p10=-20.0,
        p25=-8.0,
        p50=0.0,
        p75=10.0,
        p90=25.0,
        count=100,
        std=15.0,
        mean_offset=1.0,
    )
    bat_buckets = ResidualBuckets(buckets={"all": bat_percs}, player_type=PlayerType.BATTER)
    pitch_buckets = ResidualBuckets(buckets={"all": pitch_percs}, player_type=PlayerType.PITCHER)
    save_residual_buckets(
        {"batter": bat_buckets, "pitcher": pitch_buckets},
        artifact_dir / "pt_residual_buckets.joblib",
    )


class TestPlayingTimePredict:
    def test_predict_loads_coefficients_and_produces_predictions(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        batting_rows = [_make_batting_row(1, 2023)]
        pitching_rows = [_make_pitching_row(10, 2023)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        assert result.model_name == "playing_time"
        assert len(result.predictions) == 2

    def test_predict_clamps_batting_to_750(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        # With intercept 50 + 0.8*9999 = ~8050 -> clamped to 750
        batting_rows = [_make_batting_row(1, 2023, pa_1=9999.0)]
        assembler = FakeAssembler(batting_rows, pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        batter_preds = [p for p in result.predictions if p["player_type"] == "batter"]
        assert batter_preds[0]["pa"] <= 750

    def test_predict_clamps_to_zero(self, tmp_path: Path) -> None:
        # Fabricate a row where prediction goes negative
        row = _make_batting_row(1, 2023, pa_1=0.0)
        row["pa_2"] = 0.0
        row["pa_3"] = 0.0
        # Override intercept by using a different saved coefficients set
        batter = PlayingTimeCoefficients(
            feature_names=("pa_1",),
            coefficients=(-1.0,),
            intercept=-100.0,
            r_squared=0.5,
            player_type=PlayerType.BATTER,
        )
        pitcher = PlayingTimeCoefficients(
            feature_names=("ip_1",),
            coefficients=(0.5,),
            intercept=20.0,
            r_squared=0.5,
            player_type=PlayerType.PITCHER,
        )
        batter_curve = AgingCurve(
            peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type=PlayerType.BATTER
        )
        pitcher_curve = AgingCurve(
            peak_age=26.0, improvement_rate=0.008, decline_rate=0.007, player_type=PlayerType.PITCHER
        )
        artifact_dir = tmp_path / "playing_time" / "latest"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        save_coefficients({"batter": batter, "pitcher": pitcher}, artifact_dir / "pt_coefficients.joblib")
        save_aging_curves({"batter": batter_curve, "pitcher": pitcher_curve}, artifact_dir / "pt_aging_curves.joblib")

        assembler = FakeAssembler([row], pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        batter_preds = [p for p in result.predictions if p["player_type"] == "batter"]
        assert batter_preds[0]["pa"] == 0

    def test_predict_batter_output_format(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        batting_rows = [_make_batting_row(1, 2023)]
        assembler = FakeAssembler(batting_rows, pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        batter_preds = [p for p in result.predictions if p["player_type"] == "batter"]
        assert len(batter_preds) == 1
        pred = batter_preds[0]
        assert pred["player_id"] == 1
        assert pred["season"] == 2024
        assert "pa" in pred

    def test_predict_pitcher_output_format(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        pitching_rows = [_make_pitching_row(10, 2023)]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        pitcher_preds = [p for p in result.predictions if p["player_type"] == "pitcher"]
        assert len(pitcher_preds) == 1
        pred = pitcher_preds[0]
        assert pred["player_id"] == 10
        assert pred["season"] == 2024
        assert "ip" in pred

    def test_predict_empty_data(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        assert result.model_name == "playing_time"
        assert len(result.predictions) == 0

    def test_predict_age_affects_output(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        young_row = _make_batting_row(1, 2023)
        young_row["age"] = 24
        old_row = _make_batting_row(2, 2023)
        old_row["age"] = 35
        assembler = FakeAssembler([young_row, old_row], pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        batter_preds = {p["player_id"]: p for p in result.predictions if p["player_type"] == "batter"}
        # Young player (below peak) should get higher age_pt_factor -> higher prediction
        assert batter_preds[1]["pa"] != batter_preds[2]["pa"]

    def test_predict_produces_distributions(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        batting_rows = [_make_batting_row(1, 2023)]
        pitching_rows = [_make_pitching_row(10, 2023)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        assert result.distributions is not None
        assert len(result.distributions) == 2

    def test_predict_distribution_percentiles_ordered(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        batting_rows = [_make_batting_row(1, 2023)]
        assembler = FakeAssembler(batting_rows, pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        assert result.distributions is not None
        d = result.distributions[0]
        assert d["p10"] <= d["p25"] <= d["p50"] <= d["p75"] <= d["p90"]

    def test_predict_batter_distribution_stat_is_pa(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        batting_rows = [_make_batting_row(1, 2023)]
        assembler = FakeAssembler(batting_rows, pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        assert result.distributions is not None
        assert result.distributions[0]["stat"] == "pa"

    def test_predict_pitcher_distribution_stat_is_ip(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        pitching_rows = [_make_pitching_row(10, 2023)]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        assert result.distributions is not None
        assert result.distributions[0]["stat"] == "ip"

    def test_predict_distribution_includes_required_keys(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        batting_rows = [_make_batting_row(1, 2023)]
        assembler = FakeAssembler(batting_rows, pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        assert result.distributions is not None
        d = result.distributions[0]
        required_keys = {"player_id", "player_type", "season", "stat", "p10", "p25", "p50", "p75", "p90", "mean", "std"}
        assert required_keys.issubset(d.keys())

    def test_predict_without_residual_buckets_file_skips_distributions(self, tmp_path: Path) -> None:
        """Backward compat: missing residual buckets file → no distributions."""
        _save_test_coefficients(tmp_path)
        # Remove the residual buckets file
        rb_path = tmp_path / "playing_time" / "latest" / "pt_residual_buckets.joblib"
        rb_path.unlink()
        batting_rows = [_make_batting_row(1, 2023)]
        assembler = FakeAssembler(batting_rows, pitching_rows=[])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        assert result.distributions is None


def _make_multi_season_batting_rows(seasons: list[int], players_per_season: int = 10) -> list[dict[str, Any]]:
    """Create batting rows across multiple seasons with varying PA."""
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for i in range(players_per_season):
            rows.append(_make_batting_row(i, season, pa_1=400.0 + i * 20 + season % 10))
    return rows


def _make_multi_season_pitching_rows(seasons: list[int], players_per_season: int = 10) -> list[dict[str, Any]]:
    """Create pitching rows across multiple seasons with varying IP."""
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for i in range(players_per_season):
            rows.append(_make_pitching_row(i + 100, season, ip_1=150.0 + i * 10 + season % 10))
    return rows


def _multi_season_train_config(tmp_path: Path, seasons: list[int]) -> ModelConfig:
    return ModelConfig(seasons=seasons, artifacts_dir=str(tmp_path), model_params={"aging_min_samples": 1})


class TestPlayingTimeTrainHoldout:
    def test_train_holdout_metrics_with_multiple_seasons(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        result = model.train(_multi_season_train_config(tmp_path, seasons))
        assert "rmse_batter_holdout" in result.metrics
        assert "r_squared_batter_holdout" in result.metrics
        assert "rmse_pitcher_holdout" in result.metrics
        assert "r_squared_pitcher_holdout" in result.metrics

    def test_train_no_holdout_with_few_seasons(self, tmp_path: Path) -> None:
        seasons = [2022, 2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        result = model.train(_multi_season_train_config(tmp_path, seasons))
        assert "rmse_batter_holdout" not in result.metrics
        assert "r_squared_batter_holdout" not in result.metrics

    def test_train_coefficient_report_in_metrics(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        result = model.train(_multi_season_train_config(tmp_path, seasons))
        assert "n_batter_features" in result.metrics
        assert "n_pitcher_features" in result.metrics
        assert result.metrics["n_batter_features"] > 0
        assert result.metrics["n_pitcher_features"] > 0


class TestPlayingTimeAblate:
    def test_ablate_returns_ablation_result(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = _multi_season_train_config(tmp_path, seasons)
        result = model.ablate(config)
        assert result.model_name == "playing_time"
        assert isinstance(result.feature_impacts, dict)

    def test_ablate_feature_impacts_has_batter_and_pitcher(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = _multi_season_train_config(tmp_path, seasons)
        result = model.ablate(config)
        batter_keys = [k for k in result.feature_impacts if k.startswith("batter:")]
        pitcher_keys = [k for k in result.feature_impacts if k.startswith("pitcher:")]
        assert len(batter_keys) > 0
        assert len(pitcher_keys) > 0

    def test_ablate_requires_multiple_seasons(self, tmp_path: Path) -> None:
        seasons = [2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = _multi_season_train_config(tmp_path, seasons)
        result = model.ablate(config)
        assert len(result.feature_impacts) == 0

    def test_model_is_ablatable(self) -> None:
        assert isinstance(PlayingTimeModel(assembler=_NULL_ASSEMBLER), Ablatable)

    def test_ablate_includes_consensus_pt_group(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = _multi_season_train_config(tmp_path, seasons)
        result = model.ablate(config)
        assert "batter:consensus_pt" in result.feature_impacts
        assert "pitcher:consensus_pt" in result.feature_impacts

    def test_train_with_missing_consensus_values(self, tmp_path: Path) -> None:
        """Model handles None consensus columns gracefully (treated as 0.0)."""
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        for row in batting_rows:
            row["consensus_pa"] = None
            row["steamer_pa"] = None
            row["zips_pa"] = None
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        for row in pitching_rows:
            row["consensus_ip"] = None
            row["steamer_ip"] = None
            row["zips_ip"] = None
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        result = model.train(_train_config(tmp_path))
        assert result.metrics["r_squared_batter"] >= 0.0
        assert result.metrics["r_squared_pitcher"] >= 0.0


def _make_batting_row_three_systems(player_id: int, season: int, pa_1: float = 500.0) -> dict[str, Any]:
    """Create a batting row with three projection systems."""
    row = _make_batting_row(player_id, season, pa_1)
    row["atc_pa"] = 590.0
    row["consensus_pa"] = (row["steamer_pa"] + row["zips_pa"] + row["atc_pa"]) / 3
    return row


def _make_pitching_row_three_systems(player_id: int, season: int, ip_1: float = 180.0) -> dict[str, Any]:
    """Create a pitching row with three projection systems."""
    row = _make_pitching_row(player_id, season, ip_1)
    row["atc_ip"] = 185.0
    row["consensus_ip"] = (row["steamer_ip"] + row["zips_ip"] + row["atc_ip"]) / 3
    return row


def _make_batting_row_single_system(player_id: int, season: int, pa_1: float = 500.0) -> dict[str, Any]:
    """Create a batting row with only steamer projections."""
    row = _make_batting_row(player_id, season, pa_1)
    del row["zips_pa"]
    row["consensus_pa"] = row["steamer_pa"]
    return row


def _make_pitching_row_single_system(player_id: int, season: int, ip_1: float = 180.0) -> dict[str, Any]:
    """Create a pitching row with only steamer projections."""
    row = _make_pitching_row(player_id, season, ip_1)
    del row["zips_ip"]
    row["consensus_ip"] = row["steamer_ip"]
    return row


def _make_batting_row_autoregressive(player_id: int, season: int, pa_1: float = 500.0) -> dict[str, Any]:
    """Create a batting row with no projection systems."""
    row = _make_batting_row(player_id, season, pa_1)
    del row["steamer_pa"]
    del row["zips_pa"]
    del row["consensus_pa"]
    return row


def _make_pitching_row_autoregressive(player_id: int, season: int, ip_1: float = 180.0) -> dict[str, Any]:
    """Create a pitching row with no projection systems."""
    row = _make_pitching_row(player_id, season, ip_1)
    del row["steamer_ip"]
    del row["zips_ip"]
    del row["consensus_ip"]
    return row


class TestPlayingTimeTrainThreeSystems:
    def test_train_with_three_systems(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row_three_systems(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row_three_systems(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "pt_systems": ["steamer", "zips", "atc"]},
        )
        result = model.train(config)
        assert result.metrics["r_squared_batter"] >= 0.0
        assert result.metrics["r_squared_pitcher"] >= 0.0

    def test_train_with_three_systems_produces_predictions(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row_three_systems(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row_three_systems(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "pt_systems": ["steamer", "zips", "atc"]},
        )
        model.train(config)
        predict_config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"pt_systems": ["steamer", "zips", "atc"]},
        )
        result = model.predict(predict_config)
        assert len(result.predictions) == 20


class TestPlayingTimeTrainSingleSystem:
    def test_train_with_single_system(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row_single_system(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row_single_system(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "pt_systems": ["steamer"]},
        )
        result = model.train(config)
        assert result.metrics["r_squared_batter"] >= 0.0
        assert result.metrics["r_squared_pitcher"] >= 0.0


class TestPlayingTimeTrainAutoregressive:
    def test_train_autoregressive(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row_autoregressive(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row_autoregressive(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "pt_systems": []},
        )
        result = model.train(config)
        assert result.metrics["r_squared_batter"] >= 0.0
        assert result.metrics["r_squared_pitcher"] >= 0.0


def _make_multi_season_batting_rows_three_systems(
    seasons: list[int], players_per_season: int = 10
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for i in range(players_per_season):
            rows.append(_make_batting_row_three_systems(i, season, pa_1=400.0 + i * 20 + season % 10))
    return rows


def _make_multi_season_pitching_rows_three_systems(
    seasons: list[int], players_per_season: int = 10
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for i in range(players_per_season):
            rows.append(_make_pitching_row_three_systems(i + 100, season, ip_1=150.0 + i * 10 + season % 10))
    return rows


def _make_multi_season_batting_rows_autoregressive(
    seasons: list[int], players_per_season: int = 10
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for i in range(players_per_season):
            rows.append(_make_batting_row_autoregressive(i, season, pa_1=400.0 + i * 20 + season % 10))
    return rows


def _make_multi_season_pitching_rows_autoregressive(
    seasons: list[int], players_per_season: int = 10
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for i in range(players_per_season):
            rows.append(_make_pitching_row_autoregressive(i + 100, season, ip_1=150.0 + i * 10 + season % 10))
    return rows


class TestPlayingTimeAblateThreeSystems:
    def test_ablate_with_three_systems(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows_three_systems(seasons)
        pitching_rows = _make_multi_season_pitching_rows_three_systems(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=seasons,
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "pt_systems": ["steamer", "zips", "atc"]},
        )
        result = model.ablate(config)
        assert "batter:consensus_pt" in result.feature_impacts
        assert "pitcher:consensus_pt" in result.feature_impacts


class TestIPCalibrationIntegration:
    def test_train_with_multiple_seasons_saves_calibrator(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        model.train(_multi_season_train_config(tmp_path, seasons))
        calibrator_path = tmp_path / "playing_time" / "latest" / "pt_ip_calibrator.joblib"
        assert calibrator_path.exists()

    def test_train_single_season_no_calibrator(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        model.train(_train_config(tmp_path))
        calibrator_path = tmp_path / "playing_time" / "latest" / "pt_ip_calibrator.joblib"
        assert not calibrator_path.exists()

    def test_train_with_calibration_disabled(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows(seasons)
        pitching_rows = _make_multi_season_pitching_rows(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=seasons,
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "ip_calibration": False},
        )
        model.train(config)
        calibrator_path = tmp_path / "playing_time" / "latest" / "pt_ip_calibrator.joblib"
        assert not calibrator_path.exists()

    def test_predict_with_calibration_changes_ip(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        # Save a calibrator that shifts IP down
        cal = IPCalibrator(
            x_thresholds=(0.0, 100.0, 200.0),
            y_calibrated=(0.0, 70.0, 150.0),
        )
        artifact_dir = tmp_path / "playing_time" / "latest"
        save_ip_calibrator(cal, artifact_dir / "pt_ip_calibrator.joblib")

        pitching_rows = [_make_pitching_row(10, 2023)]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        pitcher_preds = [p for p in result.predictions if p["player_type"] == "pitcher"]
        assert len(pitcher_preds) == 1
        # With the calibrator, IP should be less than the uncalibrated value
        # ip_1=180 * 0.7 + 20 = 146 raw, calibrator maps down
        assert pitcher_preds[0]["ip"] < 200.0

    def test_predict_with_target_pitcher_count(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        cal = IPCalibrator(
            x_thresholds=(0.0, 250.0),
            y_calibrated=(0.0, 250.0),
        )
        artifact_dir = tmp_path / "playing_time" / "latest"
        save_ip_calibrator(cal, artifact_dir / "pt_ip_calibrator.joblib")

        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=50.0 + i * 20) for i in range(10)]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"target_pitcher_count": 5},
        )
        result = model.predict(config)
        pitcher_preds = [p for p in result.predictions if p["player_type"] == "pitcher"]
        assert len(pitcher_preds) == 5

    def test_predict_without_calibrator_backward_compatible(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        # No calibrator file saved — should work fine
        pitching_rows = [_make_pitching_row(10, 2023)]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        pitcher_preds = [p for p in result.predictions if p["player_type"] == "pitcher"]
        assert len(pitcher_preds) == 1
        assert pitcher_preds[0]["ip"] > 0

    def test_predict_calibration_preserves_rank_order(self, tmp_path: Path) -> None:
        _save_test_coefficients(tmp_path)
        # Calibrator that preserves order (monotone)
        cal = IPCalibrator(
            x_thresholds=(0.0, 100.0, 200.0),
            y_calibrated=(0.0, 60.0, 180.0),
        )
        artifact_dir = tmp_path / "playing_time" / "latest"
        save_ip_calibrator(cal, artifact_dir / "pt_ip_calibrator.joblib")

        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=50.0 + i * 30) for i in range(5)]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        pitcher_preds = sorted(
            [p for p in result.predictions if p["player_type"] == "pitcher"],
            key=lambda p: p["player_id"],
        )
        ips = [p["ip"] for p in pitcher_preds]
        # Monotone calibrator means higher ip_1 → higher calibrated IP
        for i in range(len(ips) - 1):
            assert ips[i] <= ips[i + 1]

    def test_calibrator_roundtrip_serialization(self, tmp_path: Path) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 50.0, 100.0, 200.0),
            y_calibrated=(0.0, 40.0, 85.0, 180.0),
        )
        path = tmp_path / "test_cal.joblib"
        save_ip_calibrator(cal, path)
        loaded = load_ip_calibrator(path)
        assert loaded.x_thresholds == cal.x_thresholds
        assert loaded.y_calibrated == cal.y_calibrated


class TestPlayingTimeAblateAutoregressive:
    def test_ablate_autoregressive(self, tmp_path: Path) -> None:
        seasons = [2020, 2021, 2022, 2023]
        batting_rows = _make_multi_season_batting_rows_autoregressive(seasons)
        pitching_rows = _make_multi_season_pitching_rows_autoregressive(seasons)
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=seasons,
            artifacts_dir=str(tmp_path),
            model_params={"aging_min_samples": 1, "pt_systems": []},
        )
        result = model.ablate(config)
        assert "batter:consensus_pt" not in result.feature_impacts
        assert "pitcher:consensus_pt" not in result.feature_impacts
        # Still has base groups
        assert "batter:base" in result.feature_impacts
        assert "pitcher:base" in result.feature_impacts


class FakeProjectionRepo:
    """Minimal projection repo for testing projection-based spine."""

    def __init__(self, projections: list[Projection]) -> None:
        self._projections = projections

    def upsert(self, projection: Projection) -> int:
        raise NotImplementedError

    def get_by_player_season(
        self,
        player_id: int,
        season: int,
        system: str | None = None,
        *,
        include_distributions: bool = False,
    ) -> list[Projection]:
        raise NotImplementedError

    def get_by_season(
        self,
        season: int,
        system: str | None = None,
        *,
        include_distributions: bool = False,
    ) -> list[Projection]:
        return [p for p in self._projections if p.season == season and (system is None or p.system == system)]

    def get_by_system_version(self, system: str, version: str) -> list[Projection]:
        raise NotImplementedError

    def delete_by_system_version(self, system: str, version: str) -> int:
        raise NotImplementedError

    def upsert_distributions(self, projection_id: int, distributions: list[Any]) -> None:
        raise NotImplementedError

    def get_distributions(self, projection_id: int) -> list[Any]:
        raise NotImplementedError


class TestProjectionBasedSpine:
    def test_get_projection_pitcher_ids_returns_union(self) -> None:
        projections = [
            Projection(
                player_id=1, season=2024, system="steamer", version="v1", player_type=PlayerType.PITCHER, stat_json={}
            ),
            Projection(
                player_id=2, season=2024, system="zips", version="v1", player_type=PlayerType.PITCHER, stat_json={}
            ),
            Projection(
                player_id=1, season=2024, system="zips", version="v1", player_type=PlayerType.PITCHER, stat_json={}
            ),
            Projection(
                player_id=3, season=2024, system="steamer", version="v1", player_type=PlayerType.BATTER, stat_json={}
            ),
        ]
        repo = FakeProjectionRepo(projections)
        model = PlayingTimeModel(assembler=_NULL_ASSEMBLER, projection_repo=repo)
        ids = model._get_projection_pitcher_ids(2024)
        assert ids == (1, 2)

    def test_get_projection_pitcher_ids_no_repo(self) -> None:
        model = PlayingTimeModel(assembler=_NULL_ASSEMBLER)
        assert model._get_projection_pitcher_ids(2024) is None

    def test_get_projection_pitcher_ids_no_data(self) -> None:
        repo = FakeProjectionRepo([])
        model = PlayingTimeModel(assembler=_NULL_ASSEMBLER, projection_repo=repo)
        assert model._get_projection_pitcher_ids(2024) is None

    def test_predict_uses_projection_spine(self, tmp_path: Path) -> None:
        """When projection_repo is provided, predict() uses projection-based pitcher IDs."""
        _save_test_coefficients(tmp_path)

        # Create projections for pitcher IDs 10 and 20 in season 2024
        projections = [
            Projection(
                player_id=10, season=2024, system="steamer", version="v1", player_type=PlayerType.PITCHER, stat_json={}
            ),
            Projection(
                player_id=20, season=2024, system="zips", version="v1", player_type=PlayerType.PITCHER, stat_json={}
            ),
        ]
        repo = FakeProjectionRepo(projections)

        # Provide rows for both players — the assembler should see these
        pitching_rows = [
            _make_pitching_row(10, 2023),
            _make_pitching_row(20, 2023),
        ]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        model = PlayingTimeModel(assembler=assembler, projection_repo=repo)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        pitcher_preds = [p for p in result.predictions if p["player_type"] == "pitcher"]
        assert len(pitcher_preds) == 2

    def test_predict_without_projection_repo_uses_stats_spine(self, tmp_path: Path) -> None:
        """Without projection_repo, predict() falls back to stats-based spine."""
        _save_test_coefficients(tmp_path)
        pitching_rows = [_make_pitching_row(10, 2023)]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        model = PlayingTimeModel(assembler=assembler)  # no projection_repo
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.predict(config)
        pitcher_preds = [p for p in result.predictions if p["player_type"] == "pitcher"]
        assert len(pitcher_preds) == 1

    def test_build_feature_sets_with_player_ids_uses_player_ids_spine(self) -> None:
        """_build_feature_sets uses player_ids spine when provided."""
        model = PlayingTimeModel(assembler=_NULL_ASSEMBLER)
        _, pitching_fs = model._build_feature_sets([2023], pitcher_player_ids=(1, 2, 3))
        assert pitching_fs.spine_filter.player_ids == (1, 2, 3)
        assert pitching_fs.spine_filter.min_ip is None

    def test_build_feature_sets_without_player_ids_uses_stats_spine(self) -> None:
        """_build_feature_sets falls back to stats spine when no player_ids."""
        model = PlayingTimeModel(assembler=_NULL_ASSEMBLER)
        _, pitching_fs = model._build_feature_sets([2023], min_ip=10.0)
        assert pitching_fs.spine_filter.player_ids is None
        assert pitching_fs.spine_filter.min_ip == 10.0

    def test_predict_consensus_fallback_for_lagless_pitcher(self, tmp_path: Path) -> None:
        """Pitchers with no lag features use consensus_ip instead of OLS prediction."""
        _save_test_coefficients(tmp_path)
        # Row with no lag features (all None) but has consensus_ip
        lagless_row = _make_pitching_row(50, 2023)
        lagless_row["ip_1"] = None
        lagless_row["ip_2"] = None
        lagless_row["ip_3"] = None
        lagless_row["consensus_ip"] = 180.0

        # Row with lag features (normal)
        normal_row = _make_pitching_row(51, 2023)

        assembler = FakeAssembler(batting_rows=[], pitching_rows=[lagless_row, normal_row])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"ip_calibration": False},
        )
        result = model.predict(config)
        preds = {p["player_id"]: p for p in result.predictions if p["player_type"] == "pitcher"}

        # Lagless pitcher should get consensus_ip (180.0)
        assert preds[50]["ip"] == pytest.approx(180.0)
        # Normal pitcher should get OLS prediction (different from consensus)
        assert preds[51]["ip"] != 180.0
        assert preds[51]["ip"] > 0

    def test_predict_lagless_pitcher_no_consensus_gets_zero(self, tmp_path: Path) -> None:
        """Pitchers with no lags and no consensus_ip get 0 IP."""
        _save_test_coefficients(tmp_path)
        row = _make_pitching_row(50, 2023)
        row["ip_1"] = None
        row["ip_2"] = None
        row["ip_3"] = None
        row["consensus_ip"] = None

        assembler = FakeAssembler(batting_rows=[], pitching_rows=[row])
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(
            seasons=[2023],
            artifacts_dir=str(tmp_path),
            model_params={"ip_calibration": False},
        )
        result = model.predict(config)
        preds = [p for p in result.predictions if p["player_type"] == "pitcher"]
        assert preds[0]["ip"] == 0.0
