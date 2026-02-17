import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import _parse_params, app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.model_run import ArtifactType, ModelRunRecord
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.models.marcel import MarcelModel
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult, TrainResult
from fantasy_baseball_manager.models.registry import _clear, register
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo

runner = CliRunner()


def _ensure_marcel_registered() -> None:
    """Clear and re-register marcel so each test starts with a known state."""
    _clear()
    register("marcel")(MarcelModel)


def _seed_batting_data(conn: sqlite3.Connection) -> None:
    """Insert minimal data for prepare integration test."""
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    batting_rows = [
        (1, 2020, "fangraphs", 250, 17, 35, 200, 56),
        (1, 2021, "fangraphs", 500, 30, 60, 420, 120),
        (1, 2022, "fangraphs", 550, 40, 70, 460, 140),
        (1, 2023, "fangraphs", 600, 35, 65, 500, 150),
    ]
    conn.executemany(
        "INSERT INTO batting_stats (player_id, season, source, pa, hr, bb, ab, h) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        batting_rows,
    )
    conn.commit()


class TestListCommand:
    def test_list_shows_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "marcel" in result.output

    def test_list_empty_registry(self) -> None:
        _clear()
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No models registered" in result.output


class TestInfoCommand:
    def test_info_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["info", "marcel"])
        assert result.exit_code == 0
        assert "marcel" in result.output
        assert "prepare" in result.output
        assert "predict" in result.output
        assert "evaluate" in result.output

    def test_info_unknown_model(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["info", "nonexistent"])
        assert result.exit_code != 0


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
            "fantasy_baseball_manager.cli.app.load_config",
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

        monkeypatch.setattr("fantasy_baseball_manager.cli.app.load_config", spy_load)
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


class TestFeaturesCommand:
    def test_features_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "marcel"])
        assert result.exit_code == 0
        assert "Features for model 'marcel'" in result.output
        assert "80 features" in result.output
        assert "hr_1" in result.output
        assert "age" in result.output

    def test_features_unknown_model(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "nonexistent"])
        assert result.exit_code != 0

    def test_features_shows_lag(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "marcel"])
        assert "lag=" in result.output

    def test_features_shows_computed(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "marcel"])
        assert "computed=age" in result.output


def _seed_player_for_import(conn: sqlite3.Connection) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, fangraphs_id=10155))


class TestImportCommand:
    def test_import_command_exists(self) -> None:
        result = runner.invoke(app, ["import", "--help"])
        assert result.exit_code == 0
        assert "third-party" in result.output.lower() or "csv" in result.output.lower()

    def test_import_batting_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        seed_conn = create_connection(db_path)
        _seed_player_for_import(seed_conn)
        seed_conn.commit()
        seed_conn.close()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        csv_file = tmp_path / "steamer_batting.csv"
        csv_file.write_text("PlayerId,MLBAMID,PA,HR,AVG,WAR\n10155,545361,600,35,0.302,8.5\n")

        result = runner.invoke(
            app,
            [
                "import",
                "steamer",
                str(csv_file),
                "--version",
                "2025.1",
                "--player-type",
                "batter",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(verify_conn)
        projections = proj_repo.get_by_season(2025, system="steamer")
        assert len(projections) == 1
        assert projections[0].source_type == "third_party"
        assert projections[0].stat_json["hr"] == 35
        verify_conn.close()

    def test_import_pitching_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        seed_conn = create_connection(db_path)
        _seed_player_for_import(seed_conn)
        seed_conn.commit()
        seed_conn.close()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        csv_file = tmp_path / "steamer_pitching.csv"
        csv_file.write_text("PlayerId,MLBAMID,W,L,ERA,SO,IP,WAR\n10155,545361,12,6,3.00,200,185.0,5.5\n")

        result = runner.invoke(
            app,
            [
                "import",
                "steamer",
                str(csv_file),
                "--version",
                "2025.1",
                "--player-type",
                "pitcher",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(verify_conn)
        projections = proj_repo.get_by_season(2025, system="steamer")
        assert len(projections) == 1
        assert projections[0].source_type == "third_party"
        assert projections[0].stat_json["era"] == 3.00
        verify_conn.close()

    def test_import_missing_file_exits_with_error(self) -> None:
        result = runner.invoke(
            app,
            [
                "import",
                "steamer",
                "/nonexistent/path.csv",
                "--version",
                "2025.1",
                "--player-type",
                "batter",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code != 0


def _seed_eval_data(conn: sqlite3.Connection, system: str = "steamer", version: str = "2025.1") -> None:
    """Seed projections and actuals for eval/compare tests."""
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (2, 'Aaron', 'Judge', '1992-04-26', 'R')"
    )
    proj_repo = SqliteProjectionRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    proj_repo.upsert(
        Projection(
            player_id=1,
            season=2025,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"hr": 30, "avg": 0.280},
            source_type="third_party",
        )
    )
    proj_repo.upsert(
        Projection(
            player_id=2,
            season=2025,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"hr": 45, "avg": 0.310},
            source_type="third_party",
        )
    )
    batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", hr=28, avg=0.265))
    batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", hr=40, avg=0.300))
    conn.commit()


class TestCompareCommand:
    def test_compare_command_exists(self) -> None:
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0

    def test_compare_with_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_eval_data(db_conn, system="steamer", version="2025.1")
        _seed_eval_data(db_conn, system="zips", version="2025.1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["compare", "steamer/2025.1", "zips/2025.1", "--season", "2025"],
        )
        assert result.exit_code == 0, result.output
        assert "steamer" in result.output
        assert "zips" in result.output


def _seed_model_run(conn: sqlite3.Connection, system: str = "marcel", version: str = "v1") -> None:
    repo = SqliteModelRunRepo(conn)
    repo.upsert(
        ModelRunRecord(
            system=system,
            version=version,
            config_json={"data_dir": "./data", "seasons": [2023]},
            artifact_type="none",
            artifact_path=None,
            git_commit="abc123",
            tags_json={"env": "test"},
            metrics_json={"rmse": 0.5},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )


class TestRunsListCommand:
    def test_runs_list_command_exists(self) -> None:
        result = runner.invoke(app, ["runs", "list", "--help"])
        assert result.exit_code == 0

    def test_runs_list_shows_records(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_model_run(db_conn, system="marcel", version="v1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "list", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "marcel" in result.output
        assert "v1" in result.output

    def test_runs_list_filter_by_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_model_run(db_conn, system="marcel", version="v1")
        _seed_model_run(db_conn, system="steamer", version="v1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "list", "--model", "marcel", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "marcel" in result.output
        assert "steamer" not in result.output

    def test_runs_list_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "list", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No runs found" in result.output


class TestRunsShowCommand:
    def test_runs_show_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_model_run(db_conn, system="marcel", version="v1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "show", "marcel/v1", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "marcel" in result.output
        assert "v1" in result.output
        assert "abc123" in result.output

    def test_runs_show_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "show", "nonexistent/v1", "--data-dir", "./data"])
        assert result.exit_code != 0


class TestRunsDeleteCommand:
    def test_runs_delete_command(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_path = tmp_path / "fbm.db"
        seed_conn = create_connection(db_path)
        _seed_model_run(seed_conn, system="marcel", version="v1")
        seed_conn.commit()
        seed_conn.close()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        result = runner.invoke(app, ["runs", "delete", "marcel/v1", "--yes", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output

        # Verify with a SECOND connection — proves deletion was committed
        verify_conn = create_connection(db_path)
        repo = SqliteModelRunRepo(verify_conn)
        assert repo.get("marcel", "v1") is None
        verify_conn.close()

    def test_runs_delete_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["runs", "delete", "nonexistent/v1", "--yes", "--data-dir", "./data"])
        assert result.exit_code != 0


def _seed_projections_data(conn: sqlite3.Connection, system: str = "steamer", version: str = "2025.1") -> None:
    """Seed player and projection data for projections commands."""
    player_repo = SqlitePlayerRepo(conn)
    proj_repo = SqliteProjectionRepo(conn)
    pid = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    proj_repo.upsert(
        Projection(
            player_id=pid,
            season=2025,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"hr": 30, "avg": 0.280},
            source_type="third_party",
        )
    )
    conn.commit()


class TestProjectionsLookupCommand:
    def test_lookup_returns_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_projections_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["projections", "lookup", "Trout", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "steamer" in result.output
        assert "hr" in result.output

    def test_lookup_system_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_projections_data(db_conn, system="steamer")
        _seed_projections_data(db_conn, system="zips")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["projections", "lookup", "Trout", "--season", "2025", "--system", "steamer", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "steamer" in result.output
        assert "zips" not in result.output

    def test_lookup_no_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["projections", "lookup", "Nobody", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No projections found" in result.output


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
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (2, 'Aaron', 'Judge', '1992-04-26', 'R')"
    )
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


class TestProjectionsSystemsCommand:
    def test_systems_lists_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_projections_data(db_conn, system="steamer")
        _seed_projections_data(db_conn, system="zips")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["projections", "systems", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "steamer" in result.output
        assert "zips" in result.output

    def test_systems_empty_season(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["projections", "systems", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No projection systems found" in result.output


def _seed_report_data(
    conn: sqlite3.Connection,
    system: str = "statcast-gbm",
    version: str = "latest",
) -> None:
    """Seed players, projections, and batting actuals for report commands."""
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (2, 'Aaron', 'Judge', '1992-04-26', 'R')"
    )
    proj_repo = SqliteProjectionRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    proj_repo.upsert(
        Projection(
            player_id=1,
            season=2025,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"avg": 0.280, "obp": 0.350},
        )
    )
    proj_repo.upsert(
        Projection(
            player_id=2,
            season=2025,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"avg": 0.300, "obp": 0.370},
        )
    )
    # Player 1 outperforms on avg, player 2 underperforms
    batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", avg=0.310, obp=0.380))
    batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", avg=0.270, obp=0.340))
    conn.commit()


class TestReportCommands:
    def test_overperformers_command_exists(self) -> None:
        result = runner.invoke(app, ["report", "overperformers", "--help"])
        assert result.exit_code == 0

    def test_underperformers_command_exists(self) -> None:
        result = runner.invoke(app, ["report", "underperformers", "--help"])
        assert result.exit_code == 0

    def test_overperformers_with_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_report_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            [
                "report",
                "overperformers",
                "statcast-gbm/latest",
                "--season",
                "2025",
                "--player-type",
                "batter",
                "--stat",
                "avg",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Overperformers" in result.output
        assert "Mike Trout" in result.output or "Trout" in result.output

    def test_underperformers_with_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_report_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            [
                "report",
                "underperformers",
                "statcast-gbm/latest",
                "--season",
                "2025",
                "--player-type",
                "batter",
                "--stat",
                "avg",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Underperformers" in result.output
        assert "Aaron Judge" in result.output or "Judge" in result.output


class TestParseParams:
    def test_parse_params_coerces_bool(self) -> None:
        result = _parse_params(["use_playing_time=false"])
        assert result == {"use_playing_time": False}

    def test_parse_params_coerces_bool_true(self) -> None:
        result = _parse_params(["use_playing_time=True"])
        assert result == {"use_playing_time": True}

    def test_parse_params_coerces_int(self) -> None:
        result = _parse_params(["lags=5"])
        assert result == {"lags": 5}

    def test_parse_params_coerces_float(self) -> None:
        result = _parse_params(["alpha=0.1"])
        assert result == {"alpha": 0.1}

    def test_parse_params_leaves_string(self) -> None:
        result = _parse_params(["name=hello"])
        assert result == {"name": "hello"}

    def test_parse_params_none_returns_none(self) -> None:
        assert _parse_params(None) is None


def _seed_valuation_data(
    conn: sqlite3.Connection,
    system: str = "zar",
    version: str = "1.0",
    player_type: str = "batter",
) -> None:
    """Seed player and valuation data for valuations commands."""
    player_repo = SqlitePlayerRepo(conn)
    val_repo = SqliteValuationRepo(conn)
    pid1 = player_repo.upsert(Player(name_first="Juan", name_last="Soto", mlbam_id=665742))
    pid2 = player_repo.upsert(Player(name_first="Aaron", name_last="Judge", mlbam_id=592450))
    val_repo.upsert(
        Valuation(
            player_id=pid1,
            season=2025,
            system=system,
            version=version,
            projection_system="steamer",
            projection_version="2025.1",
            player_type=player_type,
            position="OF",
            value=42.5,
            rank=1,
            category_scores={"hr": 2.1, "sb": 0.5},
        )
    )
    val_repo.upsert(
        Valuation(
            player_id=pid2,
            season=2025,
            system=system,
            version=version,
            projection_system="steamer",
            projection_version="2025.1",
            player_type=player_type,
            position="DH",
            value=38.0,
            rank=2,
            category_scores={"hr": 2.8, "sb": -0.3},
        )
    )
    conn.commit()


class TestValuationsLookupCommand:
    def test_lookup_returns_breakdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["valuations", "lookup", "Soto", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Juan Soto" in result.output
        assert "42.5" in result.output

    def test_lookup_system_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_data(db_conn, system="zar")
        _seed_valuation_data(db_conn, system="auction")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["valuations", "lookup", "Soto", "--season", "2025", "--system", "zar", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "zar" in result.output
        assert "auction" not in result.output

    def test_lookup_no_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["valuations", "lookup", "Nobody", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No valuations found" in result.output


class TestValuationsRankingsCommand:
    def test_rankings_shows_leaderboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["valuations", "rankings", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Juan Soto" in result.output
        assert "Aaron Judge" in result.output

    def test_rankings_filter_by_player_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_data(db_conn, player_type="batter")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["valuations", "rankings", "--season", "2025", "--player-type", "pitcher", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "No valuations found" in result.output

    def test_rankings_empty_season(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["valuations", "rankings", "--season", "2099", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No valuations found" in result.output
