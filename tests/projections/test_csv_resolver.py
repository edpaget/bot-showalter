"""Tests for CSV projection resolver."""

from pathlib import Path

import pytest

from fantasy_baseball_manager.projections.csv_resolver import CSVProjectionResolver
from fantasy_baseball_manager.projections.models import ProjectionSystem


class TestCSVProjectionResolver:
    """Tests for CSVProjectionResolver."""

    def test_resolve_returns_paths_for_existing_files(self, tmp_path: Path) -> None:
        """resolve returns batting and pitching paths when files exist."""
        batting_path = tmp_path / "steamer_2023_batting.csv"
        pitching_path = tmp_path / "steamer_2023_pitching.csv"
        batting_path.write_text("Name\n")
        pitching_path.write_text("Name\n")

        resolver = CSVProjectionResolver(data_dir=tmp_path)
        batting, pitching = resolver.resolve(ProjectionSystem.STEAMER, 2023)

        assert batting == batting_path
        assert pitching == pitching_path

    def test_resolve_raises_for_missing_batting(self, tmp_path: Path) -> None:
        """resolve raises FileNotFoundError when batting file is missing."""
        pitching_path = tmp_path / "steamer_2023_pitching.csv"
        pitching_path.write_text("Name\n")

        resolver = CSVProjectionResolver(data_dir=tmp_path)

        with pytest.raises(FileNotFoundError, match="batting"):
            resolver.resolve(ProjectionSystem.STEAMER, 2023)

    def test_resolve_raises_for_missing_pitching(self, tmp_path: Path) -> None:
        """resolve raises FileNotFoundError when pitching file is missing."""
        batting_path = tmp_path / "steamer_2023_batting.csv"
        batting_path.write_text("Name\n")

        resolver = CSVProjectionResolver(data_dir=tmp_path)

        with pytest.raises(FileNotFoundError, match="pitching"):
            resolver.resolve(ProjectionSystem.STEAMER, 2023)

    def test_resolve_handles_zips(self, tmp_path: Path) -> None:
        """resolve works with ZiPS projection system."""
        batting_path = tmp_path / "zips_2022_batting.csv"
        pitching_path = tmp_path / "zips_2022_pitching.csv"
        batting_path.write_text("Name\n")
        pitching_path.write_text("Name\n")

        resolver = CSVProjectionResolver(data_dir=tmp_path)
        batting, pitching = resolver.resolve(ProjectionSystem.ZIPS, 2022)

        assert batting == batting_path
        assert pitching == pitching_path

    def test_available_projections_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        """available_projections returns empty list when data dir doesn't exist."""
        resolver = CSVProjectionResolver(data_dir=tmp_path / "nonexistent")
        result = resolver.available_projections()

        assert result == []

    def test_available_projections_finds_projection_pairs(self, tmp_path: Path) -> None:
        """available_projections finds all valid projection pairs."""
        # Create valid pairs
        (tmp_path / "steamer_2023_batting.csv").write_text("Name\n")
        (tmp_path / "steamer_2023_pitching.csv").write_text("Name\n")
        (tmp_path / "zips_2022_batting.csv").write_text("Name\n")
        (tmp_path / "zips_2022_pitching.csv").write_text("Name\n")

        # Create incomplete pair (only batting)
        (tmp_path / "steamer_2021_batting.csv").write_text("Name\n")

        resolver = CSVProjectionResolver(data_dir=tmp_path)
        result = resolver.available_projections()

        assert len(result) == 2
        assert (ProjectionSystem.STEAMER, 2023) in result
        assert (ProjectionSystem.ZIPS, 2022) in result
        # steamer_2021 should not be included (no pitching file)
        assert (ProjectionSystem.STEAMER, 2021) not in result

    def test_available_projections_handles_zipsdc(self, tmp_path: Path) -> None:
        """available_projections handles zipsdc system name."""
        (tmp_path / "zipsdc_2023_batting.csv").write_text("Name\n")
        (tmp_path / "zipsdc_2023_pitching.csv").write_text("Name\n")

        resolver = CSVProjectionResolver(data_dir=tmp_path)
        result = resolver.available_projections()

        assert (ProjectionSystem.ZIPS_DC, 2023) in result

    def test_uses_default_data_dir(self) -> None:
        """Uses data/projections as default data directory."""
        resolver = CSVProjectionResolver()

        # Should use default path relative to project root
        assert "data/projections" in str(resolver._data_dir) or resolver._data_dir.name == "projections"
