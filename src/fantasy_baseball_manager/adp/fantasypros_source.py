"""FantasyPros historical ADP parser and data source."""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING, overload

from fantasy_baseball_manager.adp.models import ADPEntry
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from fantasy_baseball_manager.player.identity import Player

logger = logging.getLogger(__name__)


def _normalize_headers(fieldnames: Sequence[str]) -> dict[str, str]:
    """Build a mapping from original header names to lowercase versions."""
    return {name: name.strip().lower() for name in fieldnames}


class FantasyProsADPParser:
    """Parses FantasyPros ADP CSV exports.

    Expected format: comma-separated with columns including
    Rank, Player, Team, Positions, and AVG (composite ADP).
    """

    def parse(self, path: Path) -> list[ADPEntry]:
        """Parse a FantasyPros ADP CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            List of ADPEntry sorted by ADP.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"FantasyPros ADP file not found: {path}")

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []

            header_map = _normalize_headers(reader.fieldnames)
            entries: list[ADPEntry] = []

            for row in reader:
                normalized = {header_map[k]: v for k, v in row.items() if k in header_map}
                entry = self._parse_row(normalized)
                if entry is not None:
                    entries.append(entry)

        entries.sort(key=lambda e: e.adp)
        return entries

    def _parse_row(self, row: dict[str, str | None]) -> ADPEntry | None:
        """Parse a single row into an ADPEntry, or None to skip."""
        avg_str = (row.get("avg") or "").strip()
        if not avg_str:
            logger.warning("Skipping row with empty AVG: %s", row.get("player", "?"))
            return None

        try:
            adp = float(avg_str)
        except ValueError:
            logger.warning("Skipping row with non-numeric AVG '%s': %s", avg_str, row.get("player", "?"))
            return None

        name = (row.get("player") or "").strip()
        positions_raw = (row.get("positions") or "").strip()
        positions = tuple(sorted(p.strip() for p in positions_raw.split(",") if p.strip()))

        return ADPEntry(
            name=name,
            adp=adp,
            positions=positions,
        )


class FantasyProsADPDataSource:
    """DataSource[ADPEntry] adapter for FantasyPros CSV files.

    Supports only ALL_PLAYERS queries.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._parser = FantasyProsADPParser()

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[ADPEntry]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[ADPEntry]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[ADPEntry] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[ADPEntry]] | Ok[ADPEntry] | Err[DataSourceError]:
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported for FantasyPros ADP"))

        try:
            entries = self._parser.parse(self._path)
        except FileNotFoundError as e:
            return Err(DataSourceError(str(e), cause=e))

        return Ok(entries)
