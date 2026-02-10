"""Tests for ADP CSV file resolver."""

from pathlib import Path

import pytest

from fantasy_baseball_manager.adp.csv_resolver import ADPCSVResolver


class TestADPCSVResolver:
    """Tests for ADPCSVResolver."""

    def test_resolve_returns_path(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "fantasypros_2024.csv"
        csv_file.write_text("header\n")
        resolver = ADPCSVResolver(data_dir=tmp_path)
        assert resolver.resolve(2024) == csv_file

    def test_resolve_raises_for_missing_file(self, tmp_path: Path) -> None:
        resolver = ADPCSVResolver(data_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            resolver.resolve(2024)

    def test_available_years_empty_dir(self, tmp_path: Path) -> None:
        resolver = ADPCSVResolver(data_dir=tmp_path)
        assert resolver.available_years() == []

    def test_available_years_finds_files(self, tmp_path: Path) -> None:
        (tmp_path / "fantasypros_2022.csv").write_text("")
        (tmp_path / "fantasypros_2024.csv").write_text("")
        (tmp_path / "fantasypros_2023.csv").write_text("")
        resolver = ADPCSVResolver(data_dir=tmp_path)
        assert resolver.available_years() == [2022, 2023, 2024]

    def test_available_years_ignores_non_matching(self, tmp_path: Path) -> None:
        (tmp_path / "fantasypros_2024.csv").write_text("")
        (tmp_path / "other_file.csv").write_text("")
        (tmp_path / "fantasypros_bad.csv").write_text("")
        resolver = ADPCSVResolver(data_dir=tmp_path)
        assert resolver.available_years() == [2024]

    def test_available_years_nonexistent_dir(self) -> None:
        resolver = ADPCSVResolver(data_dir=Path("/nonexistent/dir"))
        assert resolver.available_years() == []
