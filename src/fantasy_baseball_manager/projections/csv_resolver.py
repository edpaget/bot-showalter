"""Path resolution for CSV projection files."""

from pathlib import Path

from fantasy_baseball_manager.projections.models import ProjectionSystem

# Map system names to ProjectionSystem enum values
_SYSTEM_MAP: dict[str, ProjectionSystem] = {
    "steamer": ProjectionSystem.STEAMER,
    "zips": ProjectionSystem.ZIPS,
    "zipsdc": ProjectionSystem.ZIPS_DC,
}


def _get_default_data_dir() -> Path:
    """Get the default data directory for projection files.

    Returns:
        Path to data/projections relative to the project root.
    """
    # Walk up from this file to find the project root
    current = Path(__file__).resolve()
    # src/fantasy_baseball_manager/projections/csv_resolver.py -> 4 levels up
    project_root = current.parent.parent.parent.parent
    return project_root / "data" / "projections"


class CSVProjectionResolver:
    """Resolves paths for CSV projection files.

    Handles file naming conventions and discovers available projection files.
    Files are expected to follow the pattern: {system}_{year}_{batting|pitching}.csv
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize the resolver.

        Args:
            data_dir: Directory containing CSV files. Defaults to data/projections/.
        """
        self._data_dir = data_dir or _get_default_data_dir()

    def resolve(self, system: ProjectionSystem, year: int) -> tuple[Path, Path]:
        """Resolve paths for a projection system and year.

        Args:
            system: The projection system (STEAMER, ZIPS, etc.).
            year: The projection year.

        Returns:
            Tuple of (batting_path, pitching_path).

        Raises:
            FileNotFoundError: If either file does not exist.
        """
        system_name = system.value
        batting_path = self._data_dir / f"{system_name}_{year}_batting.csv"
        pitching_path = self._data_dir / f"{system_name}_{year}_pitching.csv"

        if not batting_path.exists():
            raise FileNotFoundError(f"Batting projections file not found: {batting_path}")
        if not pitching_path.exists():
            raise FileNotFoundError(f"Pitching projections file not found: {pitching_path}")

        return batting_path, pitching_path

    def available_projections(self) -> list[tuple[ProjectionSystem, int]]:
        """Discover available projection files.

        Scans the data directory for valid projection file pairs.

        Returns:
            List of (system, year) tuples for available projections.
        """
        if not self._data_dir.exists():
            return []

        # Find all batting files
        batting_files: dict[tuple[str, int], Path] = {}
        pitching_files: set[tuple[str, int]] = set()

        for path in self._data_dir.glob("*_*_batting.csv"):
            parts = path.stem.rsplit("_", 2)
            if len(parts) == 3:
                system_name, year_str, _ = parts
                try:
                    year = int(year_str)
                    batting_files[(system_name, year)] = path
                except ValueError:
                    continue

        for path in self._data_dir.glob("*_*_pitching.csv"):
            parts = path.stem.rsplit("_", 2)
            if len(parts) == 3:
                system_name, year_str, _ = parts
                try:
                    year = int(year_str)
                    pitching_files.add((system_name, year))
                except ValueError:
                    continue

        # Find pairs with both batting and pitching
        result: list[tuple[ProjectionSystem, int]] = []
        for (system_name, year), _ in batting_files.items():
            if (system_name, year) in pitching_files:
                system = _SYSTEM_MAP.get(system_name)
                if system is not None:
                    result.append((system, year))

        return sorted(result, key=lambda x: (x[0].value, x[1]))
