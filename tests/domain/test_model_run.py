import pytest

from fantasy_baseball_manager.domain.model_run import ArtifactType, ModelRunRecord


class TestArtifactType:
    def test_none_value(self) -> None:
        assert ArtifactType.NONE.value == "none"

    def test_file_value(self) -> None:
        assert ArtifactType.FILE.value == "file"

    def test_directory_value(self) -> None:
        assert ArtifactType.DIRECTORY.value == "directory"

    def test_from_string(self) -> None:
        assert ArtifactType("none") is ArtifactType.NONE
        assert ArtifactType("file") is ArtifactType.FILE
        assert ArtifactType("directory") is ArtifactType.DIRECTORY


class TestModelRunRecord:
    def test_required_fields(self) -> None:
        record = ModelRunRecord(
            system="marcel",
            version="2026.1",
            config_json={"weights": [5, 4, 3]},
            artifact_type="none",
            created_at="2026-02-14T12:00:00",
        )
        assert record.system == "marcel"
        assert record.version == "2026.1"
        assert record.config_json == {"weights": [5, 4, 3]}
        assert record.artifact_type == "none"
        assert record.created_at == "2026-02-14T12:00:00"

    def test_optional_fields_default_to_none(self) -> None:
        record = ModelRunRecord(
            system="marcel",
            version="2026.1",
            config_json={},
            artifact_type="none",
            created_at="2026-02-14T12:00:00",
        )
        assert record.train_dataset_id is None
        assert record.validation_dataset_id is None
        assert record.holdout_dataset_id is None
        assert record.metrics_json is None
        assert record.artifact_path is None
        assert record.git_commit is None
        assert record.tags_json is None
        assert record.id is None

    def test_all_fields(self) -> None:
        record = ModelRunRecord(
            system="xgb-batter",
            version="v3",
            config_json={"n_estimators": 100},
            artifact_type="file",
            created_at="2026-02-14T12:00:00",
            train_dataset_id=1,
            validation_dataset_id=2,
            holdout_dataset_id=3,
            metrics_json={"rmse": 0.15, "mae": 0.10},
            artifact_path="xgb-batter/v3/model.joblib",
            git_commit="abc1234",
            tags_json={"experiment": "baseline"},
            id=42,
        )
        assert record.train_dataset_id == 1
        assert record.validation_dataset_id == 2
        assert record.holdout_dataset_id == 3
        assert record.metrics_json == {"rmse": 0.15, "mae": 0.10}
        assert record.artifact_path == "xgb-batter/v3/model.joblib"
        assert record.git_commit == "abc1234"
        assert record.tags_json == {"experiment": "baseline"}
        assert record.id == 42

    def test_frozen(self) -> None:
        record = ModelRunRecord(
            system="marcel",
            version="2026.1",
            config_json={},
            artifact_type="none",
            created_at="2026-02-14T12:00:00",
        )
        with pytest.raises(AttributeError):
            record.system = "other"  # type: ignore[misc]
