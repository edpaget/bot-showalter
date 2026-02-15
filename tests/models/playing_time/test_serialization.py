from pathlib import Path

import pytest

from fantasy_baseball_manager.models.playing_time.engine import PlayingTimeCoefficients
from fantasy_baseball_manager.models.playing_time.serialization import (
    load_coefficients,
    save_coefficients,
)


class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        batter = PlayingTimeCoefficients(
            feature_names=("pa_1", "pa_2", "age"),
            coefficients=(0.5, 0.3, -1.0),
            intercept=100.0,
            r_squared=0.85,
            player_type="batter",
        )
        pitcher = PlayingTimeCoefficients(
            feature_names=("ip_1", "ip_2", "age"),
            coefficients=(0.4, 0.2, -0.5),
            intercept=50.0,
            r_squared=0.80,
            player_type="pitcher",
        )
        coefficients = {"batter": batter, "pitcher": pitcher}
        path = tmp_path / "pt_coefficients.joblib"
        save_coefficients(coefficients, path)
        loaded = load_coefficients(path)
        assert loaded["batter"] == batter
        assert loaded["pitcher"] == pitcher

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.joblib"
        with pytest.raises(FileNotFoundError):
            load_coefficients(path)
