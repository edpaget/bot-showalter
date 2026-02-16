import sqlite3
from typing import Any

from fantasy_baseball_manager.domain.projection import Projection, StatDistribution
from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.domain.evaluation import StatMetrics, SystemMetrics
from fantasy_baseball_manager.models.marcel import MarcelModel
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    ModelConfig,
    Predictable,
    Preparable,
    Model,
    Trainable,
)


class TestMarcelModel:
    def test_is_model(self) -> None:
        assert isinstance(MarcelModel(), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(MarcelModel(), Preparable)

    def test_is_not_trainable(self) -> None:
        assert not isinstance(MarcelModel(), Trainable)

    def test_is_evaluable(self) -> None:
        assert isinstance(MarcelModel(), Evaluable)

    def test_is_predictable(self) -> None:
        assert isinstance(MarcelModel(), Predictable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(MarcelModel(), FineTunable)

    def test_artifact_type(self) -> None:
        assert MarcelModel().artifact_type == "none"

    def test_name(self) -> None:
        assert MarcelModel().name == "marcel"

    def test_supported_operations(self) -> None:
        ops = MarcelModel().supported_operations
        assert ops == frozenset({"prepare", "predict", "evaluate"})

    def test_is_evaluable_with_evaluator(self) -> None:
        metrics = SystemMetrics(system="marcel", version="latest", source_type="first_party", metrics={})
        evaluator = _FakeEvaluator(metrics)
        model = MarcelModel(evaluator=evaluator)
        assert isinstance(model, Evaluable)


class TestMarcelPrepare:
    def test_prepare_returns_result_with_row_count(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2023])
        result = MarcelModel(assembler=assembler).prepare(config)
        assert result.model_name == "marcel"
        assert result.rows_processed > 0

    def test_prepare_uses_feature_dsl(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2022, 2023])
        result = MarcelModel(assembler=assembler).prepare(config)
        assert result.rows_processed > 0
        assert result.artifacts_path == config.artifacts_dir

    def test_prepare_idempotent(self, seeded_conn: sqlite3.Connection) -> None:
        """Calling prepare twice with same config returns cached result."""
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2023])
        model = MarcelModel(assembler=assembler)
        result1 = model.prepare(config)
        result2 = model.prepare(config)
        assert result1.rows_processed == result2.rows_processed


class FakeAssembler:
    """In-memory assembler for integration testing predict()."""

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


class TestMarcelPredict:
    def test_predict_returns_result(self) -> None:
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
                "pa_3": 500,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
            {
                "player_id": 2,
                "season": 2023,
                "age": 25,
                "pa_1": 500,
                "pa_2": 450,
                "pa_3": 400,
                "hr_1": 20.0,
                "hr_2": 18.0,
                "hr_3": 15.0,
                "hr_wavg": 217.0 / 5500.0,
                "weighted_pt": 5500.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        model = MarcelModel(assembler=assembler)
        result = model.predict(config)
        assert result.model_name == "marcel"
        assert len(result.predictions) == 2

    def test_predict_with_pitchers(self) -> None:
        pitching_rows = [
            {
                "player_id": 10,
                "season": 2023,
                "age": 28,
                "ip_1": 180.0,
                "ip_2": 170.0,
                "ip_3": 160.0,
                "g_1": 30,
                "g_2": 28,
                "g_3": 26,
                "gs_1": 30,
                "gs_2": 28,
                "gs_3": 26,
                "so_1": 200.0,
                "so_2": 180.0,
                "so_3": 150.0,
                "so_wavg": 1110.0 / 1040.0,
                "weighted_pt": 1040.0,
                "league_so_rate": 200.0 / 180.0,
            },
        ]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={
                "batting_categories": ["hr"],
                "pitching_categories": ["so"],
            },
        )
        model = MarcelModel(assembler=assembler)
        result = model.predict(config)
        assert result.model_name == "marcel"
        # Should have pitcher projections
        pitcher_preds = [p for p in result.predictions if p.get("player_type") == "pitcher"]
        assert len(pitcher_preds) == 1

    def test_predict_empty_data(self) -> None:
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        model = MarcelModel(assembler=assembler)
        result = model.predict(config)
        assert result.model_name == "marcel"
        assert len(result.predictions) == 0


def _batter_row(player_id: int, position: str | None) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "season": 2023,
        "age": 29,
        "position": position,
        "pa_1": 600,
        "pa_2": 550,
        "pa_3": 500,
        "hr_1": 30.0,
        "hr_2": 25.0,
        "hr_3": 20.0,
        "hr_wavg": 310.0 / 6700.0,
        "weighted_pt": 6700.0,
        "league_hr_rate": 50.0 / 1100.0,
    }


def _pitcher_row(player_id: int, position: str | None) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "season": 2023,
        "age": 28,
        "position": position,
        "ip_1": 180.0,
        "ip_2": 170.0,
        "ip_3": 160.0,
        "g_1": 30,
        "g_2": 28,
        "g_3": 26,
        "gs_1": 30,
        "gs_2": 28,
        "gs_3": 26,
        "so_1": 200.0,
        "so_2": 180.0,
        "so_3": 150.0,
        "so_wavg": 1110.0 / 1040.0,
        "weighted_pt": 1040.0,
        "league_so_rate": 200.0 / 180.0,
    }


_PREDICT_CONFIG = ModelConfig(
    seasons=[2023],
    model_params={
        "batting_categories": ["hr"],
        "pitching_categories": ["so"],
    },
)


class TestMarcelPositionFiltering:
    def test_batter_position_included_in_batting(self) -> None:
        assembler = FakeAssembler(
            batting_rows=[_batter_row(1, "SS")],
            pitching_rows=[],
        )
        result = MarcelModel(assembler=assembler).predict(_PREDICT_CONFIG)
        batter_ids = [p["player_id"] for p in result.predictions if p["player_type"] == "batter"]
        assert 1 in batter_ids

    def test_batter_position_excluded_from_pitching(self) -> None:
        assembler = FakeAssembler(
            batting_rows=[],
            pitching_rows=[_pitcher_row(1, "SS")],
        )
        result = MarcelModel(assembler=assembler).predict(_PREDICT_CONFIG)
        pitcher_ids = [p["player_id"] for p in result.predictions if p["player_type"] == "pitcher"]
        assert 1 not in pitcher_ids

    def test_pitcher_position_included_in_pitching(self) -> None:
        assembler = FakeAssembler(
            batting_rows=[],
            pitching_rows=[_pitcher_row(10, "P")],
        )
        result = MarcelModel(assembler=assembler).predict(_PREDICT_CONFIG)
        pitcher_ids = [p["player_id"] for p in result.predictions if p["player_type"] == "pitcher"]
        assert 10 in pitcher_ids

    def test_pitcher_position_excluded_from_batting(self) -> None:
        assembler = FakeAssembler(
            batting_rows=[_batter_row(10, "P")],
            pitching_rows=[],
        )
        result = MarcelModel(assembler=assembler).predict(_PREDICT_CONFIG)
        batter_ids = [p["player_id"] for p in result.predictions if p["player_type"] == "batter"]
        assert 10 not in batter_ids

    def test_two_way_included_in_both(self) -> None:
        assembler = FakeAssembler(
            batting_rows=[_batter_row(20, "DH,P")],
            pitching_rows=[_pitcher_row(20, "DH,P")],
        )
        result = MarcelModel(assembler=assembler).predict(_PREDICT_CONFIG)
        batter_ids = [p["player_id"] for p in result.predictions if p["player_type"] == "batter"]
        pitcher_ids = [p["player_id"] for p in result.predictions if p["player_type"] == "pitcher"]
        assert 20 in batter_ids
        assert 20 in pitcher_ids

    def test_null_position_included_in_both(self) -> None:
        assembler = FakeAssembler(
            batting_rows=[_batter_row(30, None)],
            pitching_rows=[_pitcher_row(30, None)],
        )
        result = MarcelModel(assembler=assembler).predict(_PREDICT_CONFIG)
        batter_ids = [p["player_id"] for p in result.predictions if p["player_type"] == "batter"]
        pitcher_ids = [p["player_id"] for p in result.predictions if p["player_type"] == "pitcher"]
        assert 30 in batter_ids
        assert 30 in pitcher_ids


class _FakeProjectionRepo:
    def __init__(self, projections: list[Projection]) -> None:
        self._projections = projections

    def upsert(self, projection: Projection) -> int:
        return 1

    def get_by_player_season(
        self, player_id: int, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        return [p for p in self._projections if p.player_id == player_id and p.season == season]

    def get_by_season(
        self, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        return [p for p in self._projections if p.season == season and (system is None or p.system == system)]

    def get_by_system_version(self, system: str, version: str) -> list[Projection]:
        return [p for p in self._projections if p.system == system and p.version == version]

    def upsert_distributions(self, projection_id: int, distributions: list[StatDistribution]) -> None:
        pass

    def get_distributions(self, projection_id: int) -> list[StatDistribution]:
        return []


class _FakeEvaluator:
    def __init__(self, result: SystemMetrics) -> None:
        self._result = result
        self.calls: list[tuple[str, str, int, int | None]] = []

    def evaluate(
        self,
        system: str,
        version: str,
        season: int,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
    ) -> SystemMetrics:
        self.calls.append((system, version, season, top))
        return self._result


class TestMarcelEvaluate:
    def test_evaluate_delegates_to_evaluator(self) -> None:
        metrics = SystemMetrics(
            system="marcel",
            version="latest",
            source_type="first_party",
            metrics={"hr": StatMetrics(rmse=0.1, mae=0.05, correlation=0.9, n=100)},
        )
        evaluator = _FakeEvaluator(metrics)
        model = MarcelModel(evaluator=evaluator)
        config = ModelConfig(seasons=[2025])
        result = model.evaluate(config)
        assert result is metrics
        assert evaluator.calls == [("marcel", "latest", 2025, None)]

    def test_evaluate_uses_config_version(self) -> None:
        metrics = SystemMetrics(
            system="marcel",
            version="v2",
            source_type="first_party",
            metrics={},
        )
        evaluator = _FakeEvaluator(metrics)
        model = MarcelModel(evaluator=evaluator)
        config = ModelConfig(seasons=[2024], version="v2")
        result = model.evaluate(config)
        assert result is metrics
        assert evaluator.calls == [("marcel", "v2", 2024, None)]

    def test_evaluate_passes_top(self) -> None:
        metrics = SystemMetrics(
            system="marcel",
            version="latest",
            source_type="first_party",
            metrics={},
        )
        evaluator = _FakeEvaluator(metrics)
        model = MarcelModel(evaluator=evaluator)
        config = ModelConfig(seasons=[2025], top=300)
        result = model.evaluate(config)
        assert result is metrics
        assert evaluator.calls == [("marcel", "latest", 2025, 300)]


class TestMarcelPredictWithProjectionRepo:
    def test_uses_playing_time_pt(self) -> None:
        """When projection_repo has PT projections, Marcel uses them."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
                "pa_3": 500,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        pt_projection = Projection(
            player_id=1,
            season=2024,
            system="playing_time",
            version="latest",
            player_type="batter",
            stat_json={"pa": 450},
        )
        assembler = FakeAssembler(batting_rows)
        repo = _FakeProjectionRepo([pt_projection])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        model = MarcelModel(assembler=assembler, projection_repo=repo)
        result = model.predict(config)
        assert len(result.predictions) == 1
        assert result.predictions[0]["pa"] == 450

    def test_without_projection_repo_uses_internal_formula(self) -> None:
        """Without a projection_repo, Marcel uses its built-in PT formula."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
                "pa_3": 500,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        model = MarcelModel(assembler=assembler)
        result = model.predict(config)
        assert len(result.predictions) == 1
        # Marcel formula: 0.5*600 + 0.1*550 + 200 = 555
        assert result.predictions[0]["pa"] == 555

    def test_partial_pt_coverage(self) -> None:
        """Repo has PT for player 1 but not player 2; player 2 falls back."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
                "pa_3": 500,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
            {
                "player_id": 2,
                "season": 2023,
                "age": 25,
                "pa_1": 500,
                "pa_2": 450,
                "pa_3": 400,
                "hr_1": 20.0,
                "hr_2": 18.0,
                "hr_3": 15.0,
                "hr_wavg": 217.0 / 5500.0,
                "weighted_pt": 5500.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        pt_projection = Projection(
            player_id=1,
            season=2024,
            system="playing_time",
            version="latest",
            player_type="batter",
            stat_json={"pa": 400},
        )
        assembler = FakeAssembler(batting_rows)
        repo = _FakeProjectionRepo([pt_projection])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        model = MarcelModel(assembler=assembler, projection_repo=repo)
        result = model.predict(config)
        by_id = {p["player_id"]: p for p in result.predictions}
        # Player 1 uses repo PT
        assert by_id[1]["pa"] == 400
        # Player 2 falls back: 0.5*500 + 0.1*450 + 200 = 495
        assert by_id[2]["pa"] == 495

    def test_two_way_player_uses_correct_pt_per_type(self) -> None:
        """A two-way player should use PA for batting and IP for pitching, not mix them."""
        batting_rows = [_batter_row(20, "DH,P")]
        pitching_rows = [_pitcher_row(20, "DH,P")]
        pt_batter = Projection(
            player_id=20,
            season=2024,
            system="playing_time",
            version="latest",
            player_type="batter",
            stat_json={"pa": 500},
        )
        pt_pitcher = Projection(
            player_id=20,
            season=2024,
            system="playing_time",
            version="latest",
            player_type="pitcher",
            stat_json={"ip": 80.0},
        )
        assembler = FakeAssembler(batting_rows, pitching_rows)
        repo = _FakeProjectionRepo([pt_batter, pt_pitcher])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"], "pitching_categories": ["so"]},
        )
        model = MarcelModel(assembler=assembler, projection_repo=repo)
        result = model.predict(config)
        batter_pred = [p for p in result.predictions if p["player_type"] == "batter"][0]
        pitcher_pred = [p for p in result.predictions if p["player_type"] == "pitcher"][0]
        assert batter_pred["pa"] == 500
        assert pitcher_pred["ip"] == 80.0

    def test_use_playing_time_false_skips_repo(self) -> None:
        """Setting use_playing_time=False ignores the projection_repo."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
                "pa_3": 500,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        pt_projection = Projection(
            player_id=1,
            season=2024,
            system="playing_time",
            version="latest",
            player_type="batter",
            stat_json={"pa": 450},
        )
        assembler = FakeAssembler(batting_rows)
        repo = _FakeProjectionRepo([pt_projection])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"], "use_playing_time": False},
        )
        model = MarcelModel(assembler=assembler, projection_repo=repo)
        result = model.predict(config)
        # Should use Marcel formula (0.5*600 + 0.1*550 + 200 = 555), NOT repo's 450
        assert result.predictions[0]["pa"] == 555


class TestMarcelPredictWithStatcastAugment:
    def test_statcast_augment_adjusts_rates(self) -> None:
        """With statcast_augment=True, Marcel's rates get blended with statcast estimates."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
                "pa_3": 500,
                "h_1": 160.0,
                "h_2": 140.0,
                "h_3": 130.0,
                "h_wavg": (160 * 5.0 + 140 * 4.0 + 130 * 3.0) / (600 * 5.0 + 550 * 4.0 + 500 * 3.0),
                "weighted_pt": 600 * 5.0 + 550 * 4.0 + 500 * 3.0,
                "league_h_rate": 0.060,
            },
        ]
        # Statcast projection for prior season (2023) with higher avg
        sc_projection = Projection(
            player_id=1,
            season=2023,
            system="statcast-gbm",
            version="latest",
            player_type="batter",
            stat_json={"avg": 0.300, "obp": 0.370, "slg": 0.480},
        )
        assembler = FakeAssembler(batting_rows)
        repo = _FakeProjectionRepo([sc_projection])
        config_with = ModelConfig(
            seasons=[2023],
            model_params={
                "batting_categories": ["h"],
                "use_playing_time": False,
                "statcast_augment": True,
                "statcast_system": "statcast-gbm",
                "statcast_version": "latest",
                "statcast_weight": 0.3,
            },
        )
        config_without = ModelConfig(
            seasons=[2023],
            model_params={
                "batting_categories": ["h"],
                "use_playing_time": False,
            },
        )
        model_with = MarcelModel(assembler=assembler, projection_repo=repo)
        model_without = MarcelModel(assembler=assembler)

        result_with = model_with.predict(config_with)
        result_without = model_without.predict(config_without)

        # Both should produce one prediction
        assert len(result_with.predictions) == 1
        assert len(result_without.predictions) == 1

        # Statcast has avg=0.300 which is higher than the weighted average
        # So the augmented prediction should have more hits
        assert result_with.predictions[0]["h"] > result_without.predictions[0]["h"]

    def test_statcast_augment_disabled_by_default(self) -> None:
        """Without statcast_augment param, no augmentation occurs."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 0,
                "pa_3": 0,
                "h_1": 160.0,
                "h_2": 0.0,
                "h_3": 0.0,
                "h_wavg": 160 / 600,
                "weighted_pt": 600 * 5.0,
                "league_h_rate": 0.060,
            },
        ]
        sc_projection = Projection(
            player_id=1,
            season=2023,
            system="statcast-gbm",
            version="latest",
            player_type="batter",
            stat_json={"avg": 0.400},
        )
        assembler = FakeAssembler(batting_rows)
        repo = _FakeProjectionRepo([sc_projection])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["h"], "use_playing_time": False},
        )
        model = MarcelModel(assembler=assembler, projection_repo=repo)
        result = model.predict(config)
        # Despite statcast saying avg=0.400, the result should NOT be affected
        # since statcast_augment is not enabled
        assert len(result.predictions) == 1
