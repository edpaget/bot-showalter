from pathlib import Path

from fantasy_baseball_manager.models.statcast_gbm.serialization import load_models, save_models


class TestSerialization:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        models = {"model_a": {"weight": 1.0}, "model_b": [1, 2, 3]}
        path = tmp_path / "models.joblib"
        save_models(models, path)
        loaded = load_models(path)
        assert loaded == models
