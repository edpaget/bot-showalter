from pathlib import Path
from typing import Any

from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.playing_time.engine import PlayingTimeCoefficients
from fantasy_baseball_manager.models.playing_time.model import PlayingTimeModel
from fantasy_baseball_manager.models.playing_time.serialization import save_coefficients
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Preparable,
    Trainable,
)


class FakeAssembler:
    """In-memory assembler for testing."""

    def __init__(self, batting_rows: list[dict[str, Any]], pitching_rows: list[dict[str, Any]] | None = None) -> None:
        self._batting_rows = batting_rows
        self._pitching_rows = pitching_rows or []
        self._next_id = 1

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.get_or_materialize(feature_set)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        if "pitching" in feature_set.name:
            rows = self._pitching_rows
        else:
            rows = self._batting_rows
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
        # Training target:
        "target_ip": ip_1 + 5.0,
    }


class TestPlayingTimeModelProtocol:
    def test_is_model(self) -> None:
        assert isinstance(PlayingTimeModel(), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(PlayingTimeModel(), Preparable)

    def test_is_predictable(self) -> None:
        assert isinstance(PlayingTimeModel(), Predictable)

    def test_is_trainable(self) -> None:
        assert isinstance(PlayingTimeModel(), Trainable)

    def test_is_not_evaluable(self) -> None:
        assert not isinstance(PlayingTimeModel(), Evaluable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(PlayingTimeModel(), FineTunable)

    def test_name(self) -> None:
        assert PlayingTimeModel().name == "playing_time"

    def test_supported_operations_includes_train(self) -> None:
        ops = PlayingTimeModel().supported_operations
        assert ops == frozenset({"prepare", "train", "predict"})

    def test_artifact_type_is_file(self) -> None:
        assert PlayingTimeModel().artifact_type == "file"


class TestPlayingTimeTrain:
    def test_train_produces_coefficients_file(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        model.train(config)
        artifact = tmp_path / "playing_time" / "latest" / "pt_coefficients.joblib"
        assert artifact.exists()

    def test_train_returns_r_squared_metrics(self, tmp_path: Path) -> None:
        batting_rows = [_make_batting_row(i, 2023, pa_1=400.0 + i * 20) for i in range(10)]
        pitching_rows = [_make_pitching_row(i + 100, 2023, ip_1=150.0 + i * 10) for i in range(10)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        model = PlayingTimeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], artifacts_dir=str(tmp_path))
        result = model.train(config)
        assert "r_squared_batter" in result.metrics
        assert "r_squared_pitcher" in result.metrics
        assert result.metrics["r_squared_batter"] >= 0.0
        assert result.metrics["r_squared_pitcher"] >= 0.0


def _save_test_coefficients(tmp_path: Path) -> None:
    """Save known coefficients for predict tests."""
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
        ),
        coefficients=(0.0, 0.8, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        intercept=50.0,
        r_squared=0.9,
        player_type="batter",
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
        ),
        coefficients=(0.0, 0.7, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        intercept=20.0,
        r_squared=0.85,
        player_type="pitcher",
    )
    artifact_dir = tmp_path / "playing_time" / "latest"
    artifact_dir.mkdir(parents=True)
    save_coefficients({"batter": batter, "pitcher": pitcher}, artifact_dir / "pt_coefficients.joblib")


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
        _save_test_coefficients(tmp_path)
        # Fabricate a row where prediction goes negative
        row = _make_batting_row(1, 2023, pa_1=0.0)
        row["pa_2"] = 0.0
        row["pa_3"] = 0.0
        # intercept=50 + 0.8*0 + 0.1*0 + 0.05*0 = 50 — not negative.
        # Override intercept by using a different saved coefficients set
        batter = PlayingTimeCoefficients(
            feature_names=("pa_1",),
            coefficients=(-1.0,),
            intercept=-100.0,
            r_squared=0.5,
            player_type="batter",
        )
        pitcher = PlayingTimeCoefficients(
            feature_names=("ip_1",),
            coefficients=(0.5,),
            intercept=20.0,
            r_squared=0.5,
            player_type="pitcher",
        )
        artifact_dir = tmp_path / "playing_time" / "latest"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        save_coefficients({"batter": batter, "pitcher": pitcher}, artifact_dir / "pt_coefficients.joblib")

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
