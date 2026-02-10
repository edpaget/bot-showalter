"""Path resolution for FantasyPros ADP CSV files."""

from __future__ import annotations

from pathlib import Path


def _get_default_data_dir() -> Path:
    """Get the default data directory for ADP files.

    Returns:
        Path to data/adp relative to the project root.
    """
    current = Path(__file__).resolve()
    # src/fantasy_baseball_manager/adp/csv_resolver.py -> 4 levels up
    project_root = current.parent.parent.parent.parent
    return project_root / "data" / "adp"


class ADPCSVResolver:
    """Resolves paths for FantasyPros ADP CSV files.

    Files are expected to follow the pattern: fantasypros_{year}.csv
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize the resolver.

        Args:
            data_dir: Directory containing CSV files. Defaults to data/adp/.
        """
        self._data_dir = data_dir or _get_default_data_dir()

    def resolve(self, year: int) -> Path:
        """Resolve path for a given year.

        Args:
            year: The ADP year.

        Returns:
            Path to the CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = self._data_dir / f"fantasypros_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"FantasyPros ADP file not found: {path}")
        return path

    def available_years(self) -> list[int]:
        """Discover available ADP years.

        Returns:
            Sorted list of years for which CSV files exist.
        """
        if not self._data_dir.exists():
            return []

        years: list[int] = []
        for path in self._data_dir.glob("fantasypros_*.csv"):
            stem = path.stem  # fantasypros_2024
            parts = stem.split("_", 1)
            if len(parts) == 2:
                try:
                    years.append(int(parts[1]))
                except ValueError:
                    continue

        return sorted(years)
