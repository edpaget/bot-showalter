from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.models.marcel import MarcelModel
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult, TrainResult
from fantasy_baseball_manager.models.registry import _clear, register
from fantasy_baseball_manager.models.zar.model import ZarModel
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from tests.helpers import seed_player

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def _ensure_marcel_registered() -> None:
    """Clear and re-register marcel so each test starts with a known state."""
    _clear()
    register("marcel")(MarcelModel)


def _seed_batting_data(conn: object) -> None:
    """Insert minimal data for prepare integration test."""
    conn.execute(  # type: ignore[union-attr]
        "INSERT INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    batting_rows = [
        (1, 2020, "fangraphs", 250, 17, 35, 200, 56),
        (1, 2021, "fangraphs", 500, 30, 60, 420, 120),
        (1, 2022, "fangraphs", 550, 40, 70, 460, 140),
        (1, 2023, "fangraphs", 600, 35, 65, 500, 150),
    ]
    conn.executemany(  # type: ignore[union-attr]
        "INSERT INTO batting_stats (player_id, season, source, pa, hr, bb, ab, h) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        batting_rows,
    )
    conn.commit()  # type: ignore[union-attr]


def _eval_league() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=260,
        roster_batters=2,
        roster_pitchers=1,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=1,
        positions={"of": 1},
    )


class TestActionCommands:
    def test_train_marcel_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        result = runner.invoke(app, ["train", "marcel"])
        assert result.exit_code != 0
        assert "does not support" in result.output.lower()

    def test_prepare_marcel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        seeded_conn = create_connection(":memory:")
        _seed_batting_data(seeded_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: seeded_conn,
        )
        result = runner.invoke(app, ["prepare", "marcel", "--season", "2023"])
        assert result.exit_code == 0

    def test_evaluate_marcel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        result = runner.invoke(app, ["evaluate", "marcel", "--season", "2025"])
        assert result.exit_code == 0

    def test_finetune_marcel_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        result = runner.invoke(app, ["finetune", "marcel"])
        assert result.exit_code != 0
        assert "does not support" in result.output.lower()

    def test_predict_marcel_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        seeded_conn = create_connection(":memory:")
        _seed_batting_data(seeded_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: seeded_conn,
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.model.load_config",
            lambda **kwargs: ModelConfig(
                seasons=[2023],
                model_params={
                    "batting_categories": ["hr", "bb", "h"],
                    "pitching_categories": ["so"],
                },
            ),
        )
        result = runner.invoke(app, ["predict", "marcel", "--season", "2023"])
        assert result.exit_code == 0, result.output

    def test_ablate_marcel_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        result = runner.invoke(app, ["ablate", "marcel"])
        assert result.exit_code != 0

    def test_ablate_passes_param_to_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        captured: dict[str, object] = {}
        original_load = __import__("fantasy_baseball_manager.config", fromlist=["load_config"]).load_config

        def spy_load(**kwargs: object) -> object:
            captured.update(kwargs)
            return original_load(**kwargs)

        monkeypatch.setattr("fantasy_baseball_manager.cli.commands.model.load_config", spy_load)
        runner.invoke(app, ["ablate", "marcel", "--param", "mode=preseason"])
        assert captured.get("model_params") == {"mode": "preseason"}

    def test_train_marcel_with_output_dir_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        result = runner.invoke(app, ["train", "marcel", "--output-dir", "/tmp/out"])
        assert result.exit_code != 0

    def test_train_marcel_with_seasons_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        result = runner.invoke(app, ["train", "marcel", "--season", "2023", "--season", "2024"])
        assert result.exit_code != 0


def _register_trainable_stub() -> None:
    """Register a minimal trainable stub model for run tracking tests."""
    _clear()

    class _TrainableStub:
        @property
        def name(self) -> str:
            return "stub"

        @property
        def description(self) -> str:
            return "Stub model for testing."

        @property
        def supported_operations(self) -> frozenset[str]:
            return frozenset({"train"})

        @property
        def artifact_type(self) -> str:
            return ArtifactType.NONE.value

        def train(self, config: ModelConfig) -> TrainResult:
            return TrainResult(model_name="stub", metrics={}, artifacts_path=config.artifacts_dir)

    register("stub")(_TrainableStub)


class TestTrainRunTracking:
    def test_train_with_version_records_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _register_trainable_stub()
        db_path = tmp_path / "fbm.db"
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["train", "stub", "--version", "v1"])
        assert result.exit_code == 0, result.output

        # Verify with a SECOND connection — proves data was committed
        verify_conn = create_connection(db_path)
        repo = SqliteModelRunRepo(verify_conn)
        runs = repo.list(system="stub")
        assert len(runs) == 1
        assert runs[0].version == "v1"
        verify_conn.close()

    def test_train_with_tag_records_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _register_trainable_stub()
        db_path = tmp_path / "fbm.db"
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["train", "stub", "--version", "v1", "--tag", "env=test"])
        assert result.exit_code == 0, result.output

        # Verify with a SECOND connection — proves data was committed
        verify_conn = create_connection(db_path)
        repo = SqliteModelRunRepo(verify_conn)
        runs = repo.list(system="stub")
        assert len(runs) == 1
        assert runs[0].tags_json == {"env": "test"}
        verify_conn.close()

    def test_train_without_version_skips_run_tracking(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _register_trainable_stub()
        db_path = tmp_path / "fbm.db"
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["train", "stub"])
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteModelRunRepo(verify_conn)
        runs = repo.list()
        assert len(runs) == 0
        verify_conn.close()


def _register_predictable_stub() -> None:
    """Register a minimal predictable stub model for predict run tracking tests."""
    _clear()

    class _PredictableStub:
        @property
        def name(self) -> str:
            return "pred-stub"

        @property
        def description(self) -> str:
            return "Predictable stub model for testing."

        @property
        def supported_operations(self) -> frozenset[str]:
            return frozenset({"predict"})

        @property
        def artifact_type(self) -> str:
            return ArtifactType.NONE.value

        def predict(self, config: ModelConfig) -> PredictResult:
            return PredictResult(model_name="pred-stub", predictions=[{"hr": 30}], output_path=config.artifacts_dir)

    register("pred-stub")(_PredictableStub)


class TestPredictRunTracking:
    def test_predict_with_version_records_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _register_predictable_stub()
        db_path = tmp_path / "fbm.db"
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["predict", "pred-stub", "--version", "v1"])
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteModelRunRepo(verify_conn)
        runs = repo.list(system="pred-stub")
        assert len(runs) == 1
        assert runs[0].version == "v1"
        assert runs[0].operation == "predict"
        verify_conn.close()

    def test_predict_without_version_skips_run_tracking(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _register_predictable_stub()
        db_path = tmp_path / "fbm.db"
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["predict", "pred-stub"])
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        repo = SqliteModelRunRepo(verify_conn)
        runs = repo.list()
        assert len(runs) == 0
        verify_conn.close()


def _register_predictable_stub_with_projections() -> None:
    """Register a stub that returns predictions with player_id/season for persistence tests."""
    _clear()

    class _PredictableStubWithProjections:
        @property
        def name(self) -> str:
            return "pred-proj-stub"

        @property
        def description(self) -> str:
            return "Predictable stub that returns full predictions."

        @property
        def supported_operations(self) -> frozenset[str]:
            return frozenset({"predict"})

        @property
        def artifact_type(self) -> str:
            return ArtifactType.NONE.value

        def predict(self, config: ModelConfig) -> PredictResult:
            return PredictResult(
                model_name="pred-proj-stub",
                predictions=[
                    {"player_id": 1, "season": 2025, "player_type": "batter", "hr": 30, "pa": 600},
                    {"player_id": 2, "season": 2025, "player_type": "pitcher", "so": 200, "ip": 180.0},
                ],
                output_path=config.artifacts_dir,
            )

    register("pred-proj-stub")(_PredictableStubWithProjections)


def _seed_players_for_persistence(db_path: Path) -> None:
    """Seed the two players that the predictable stub references."""
    conn = create_connection(db_path)
    seed_player(conn, player_id=1, name_first="Mike", name_last="Trout", birth_date="1991-08-07")
    seed_player(conn, player_id=2, name_first="Aaron", name_last="Judge", birth_date="1992-04-26")
    conn.commit()
    conn.close()


class TestPredictPersistence:
    def test_predict_stores_projections_in_db(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _register_predictable_stub_with_projections()
        db_path = tmp_path / "fbm.db"
        _seed_players_for_persistence(db_path)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["predict", "pred-proj-stub", "--version", "v1"])
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(verify_conn)
        projections = proj_repo.get_by_season(2025, system="pred-proj-stub")
        assert len(projections) == 2
        batter_proj = [p for p in projections if p.player_type == "batter"][0]
        assert batter_proj.stat_json["hr"] == 30
        assert batter_proj.stat_json["pa"] == 600
        pitcher_proj = [p for p in projections if p.player_type == "pitcher"][0]
        assert pitcher_proj.stat_json["so"] == 200
        verify_conn.close()

    def test_predict_uses_model_name_as_system(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _register_predictable_stub_with_projections()
        db_path = tmp_path / "fbm.db"
        _seed_players_for_persistence(db_path)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["predict", "pred-proj-stub", "--version", "v1"])
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(verify_conn)
        projections = proj_repo.get_by_season(2025, system="pred-proj-stub")
        assert all(p.system == "pred-proj-stub" for p in projections)
        verify_conn.close()


class TestPredictLeagueResolution:
    def test_predict_resolves_league_param_to_league_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The predict command should resolve a league string param via load_league."""
        captured_config: list[ModelConfig] = []

        def fake_predict(self: object, config: ModelConfig) -> PredictResult:
            captured_config.append(config)
            return PredictResult(model_name="zar", predictions=[], output_path="")

        _clear()
        register("zar")(ZarModel)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.model.load_league",
            lambda name, path: _eval_league(),
        )
        monkeypatch.setattr(ZarModel, "predict", fake_predict)

        result = runner.invoke(
            app,
            ["predict", "zar", "--season", "2025", "--param", "league=h2h", "--param", "projection_system=steamer"],
        )
        assert result.exit_code == 0, result.output
        assert len(captured_config) == 1
        league = captured_config[0].model_params["league"]
        assert isinstance(league, LeagueSettings)
        assert league.name == "Test League"

    def test_predict_without_league_param_does_not_call_load_league(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no league param is provided, load_league should not be called."""
        _ensure_marcel_registered()
        seeded_conn = create_connection(":memory:")
        _seed_batting_data(seeded_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: seeded_conn,
        )

        def load_league_should_not_be_called(name: str, path: Path) -> LeagueSettings:
            raise AssertionError("load_league should not be called when no league param is set")

        monkeypatch.setattr("fantasy_baseball_manager.cli.commands.model.load_league", load_league_should_not_be_called)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.model.load_config",
            lambda **kwargs: ModelConfig(
                seasons=[2023],
                model_params={
                    "batting_categories": ["hr", "bb", "h"],
                    "pitching_categories": ["so"],
                },
            ),
        )
        result = runner.invoke(app, ["predict", "marcel", "--season", "2023"])
        assert result.exit_code == 0, result.output
