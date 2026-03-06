"""Tests for training metadata save/load/validate."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from fantasy_baseball_manager.models.training_metadata import (
    TrainingMetadata,
    load_training_metadata,
    save_training_metadata,
    validate_no_leakage,
)


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        save_training_metadata(tmp_path, [2018, 2019, 2020], [2021])
        result = load_training_metadata(tmp_path)
        assert result == TrainingMetadata(train_seasons=[2018, 2019, 2020], holdout_seasons=[2021])

    def test_round_trip_empty_holdout(self, tmp_path: Path) -> None:
        save_training_metadata(tmp_path, [2018, 2019], [])
        result = load_training_metadata(tmp_path)
        assert result == TrainingMetadata(train_seasons=[2018, 2019], holdout_seasons=[])

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_training_metadata(tmp_path) is None

    def test_seasons_are_sorted(self, tmp_path: Path) -> None:
        save_training_metadata(tmp_path, [2021, 2019, 2020], [2023, 2022])
        result = load_training_metadata(tmp_path)
        assert result is not None
        assert result.train_seasons == [2019, 2020, 2021]
        assert result.holdout_seasons == [2022, 2023]


class TestValidateNoLeakage:
    def test_raises_on_train_overlap(self, tmp_path: Path) -> None:
        save_training_metadata(tmp_path, [2018, 2019, 2020], [2021])
        with pytest.raises(ValueError, match="Data leakage.*2020"):
            validate_no_leakage(tmp_path, [2020])

    def test_raises_on_holdout_overlap(self, tmp_path: Path) -> None:
        save_training_metadata(tmp_path, [2018, 2019, 2020], [2021])
        with pytest.raises(ValueError, match="Data leakage.*2021"):
            validate_no_leakage(tmp_path, [2021])

    def test_passes_no_overlap(self, tmp_path: Path) -> None:
        save_training_metadata(tmp_path, [2018, 2019, 2020], [2021])
        validate_no_leakage(tmp_path, [2022])

    def test_legacy_no_metadata_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            validate_no_leakage(tmp_path, [2020])
        assert "No training metadata found" in caplog.text

    def test_empty_prediction_seasons(self, tmp_path: Path) -> None:
        save_training_metadata(tmp_path, [2018, 2019], [2020])
        validate_no_leakage(tmp_path, [])
