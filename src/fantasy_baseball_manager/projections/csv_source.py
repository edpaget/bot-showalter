"""CSV-based projection source for historical FanGraphs projections."""

import csv
from datetime import UTC, datetime
from pathlib import Path

from fantasy_baseball_manager.projections.models import (
    BattingProjection,
    PitchingProjection,
    ProjectionData,
    ProjectionSystem,
)


def _normalize_headers(reader: csv.DictReader) -> list[dict[str, str]]:
    """Read all rows with case-insensitive header normalization."""
    if reader.fieldnames is None:
        return []
    lower_map = {name: name.strip().lower().lstrip("\ufeff") for name in reader.fieldnames}
    rows: list[dict[str, str]] = []
    for row in reader:
        normalized = {lower_map[k]: v for k, v in row.items() if k in lower_map}
        rows.append(normalized)
    return rows


def _get_player_id(row: dict[str, str]) -> str:
    """Extract player ID from row, handling idfg/playerid variants."""
    return row.get("playerid") or row.get("idfg") or ""


def _get_mlbam_id(row: dict[str, str]) -> str | None:
    """Extract MLBAM ID from row, handling mlbamid/xmlbamid variants."""
    mlbam = row.get("mlbamid") or row.get("xmlbamid")
    return str(mlbam) if mlbam else None


def _get_int(row: dict[str, str], key: str, default: int = 0) -> int:
    """Get integer value from row with default."""
    val = row.get(key, "")
    if not val:
        return default
    return int(float(val))


def _get_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    """Get float value from row with default."""
    val = row.get(key, "")
    if not val:
        return default
    return float(val)


class CSVProjectionSource:
    """Reads projections from local CSV files.

    Supports FanGraphs CSV exports for historical Steamer/ZiPS projections.
    Handles column name variations (idfg/playerid, mlbamid/xmlbamid) and
    computes derived stats (singles, ERA, WHIP) when needed.
    """

    def __init__(
        self,
        batting_path: Path,
        pitching_path: Path,
        system: ProjectionSystem,
    ) -> None:
        """Initialize the CSV projection source.

        Args:
            batting_path: Path to batting projections CSV file.
            pitching_path: Path to pitching projections CSV file.
            system: The projection system (STEAMER, ZIPS, etc.).
        """
        self._batting_path = batting_path
        self._pitching_path = pitching_path
        self._system = system

    def fetch_projections(self) -> ProjectionData:
        """Fetch projections from CSV files.

        Returns:
            ProjectionData containing batting and pitching projections.

        Raises:
            FileNotFoundError: If a CSV file does not exist.
        """
        batting = self._parse_batting()
        pitching = self._parse_pitching()

        return ProjectionData(
            batting=tuple(batting),
            pitching=tuple(pitching),
            system=self._system,
            fetched_at=datetime.now(UTC),
        )

    def _parse_batting(self) -> list[BattingProjection]:
        """Parse batting projections from CSV."""
        with open(self._batting_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = _normalize_headers(reader)

        projections: list[BattingProjection] = []
        for row in rows:
            h = _get_int(row, "h")
            doubles = _get_int(row, "2b")
            triples = _get_int(row, "3b")
            hr = _get_int(row, "hr")

            # Compute singles if not provided
            singles_val = row.get("1b", "")
            singles = int(float(singles_val)) if singles_val else h - doubles - triples - hr

            projections.append(
                BattingProjection(
                    player_id=_get_player_id(row),
                    mlbam_id=_get_mlbam_id(row),
                    name=row.get("name", ""),
                    team=row.get("team", ""),
                    position=row.get("minpos", ""),
                    g=_get_int(row, "g"),
                    pa=_get_int(row, "pa"),
                    ab=_get_int(row, "ab"),
                    h=h,
                    singles=singles,
                    doubles=doubles,
                    triples=triples,
                    hr=hr,
                    r=_get_int(row, "r"),
                    rbi=_get_int(row, "rbi"),
                    sb=_get_int(row, "sb"),
                    cs=_get_int(row, "cs"),
                    bb=_get_int(row, "bb"),
                    so=_get_int(row, "so"),
                    hbp=_get_int(row, "hbp"),
                    sf=_get_int(row, "sf"),
                    sh=_get_int(row, "sh"),
                    obp=_get_float(row, "obp"),
                    slg=_get_float(row, "slg"),
                    ops=_get_float(row, "ops"),
                    woba=_get_float(row, "woba"),
                    war=_get_float(row, "war"),
                )
            )
        return projections

    def _parse_pitching(self) -> list[PitchingProjection]:
        """Parse pitching projections from CSV."""
        with open(self._pitching_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = _normalize_headers(reader)

        projections: list[PitchingProjection] = []
        for row in rows:
            ip = _get_float(row, "ip")
            er = _get_int(row, "er")
            h = _get_int(row, "h")
            bb = _get_int(row, "bb")

            # Compute ERA and WHIP if not provided
            era_val = row.get("era", "")
            if era_val:
                era = float(era_val)
            elif ip > 0:
                era = (er / ip) * 9
            else:
                era = 0.0

            whip_val = row.get("whip", "")
            if whip_val:
                whip = float(whip_val)
            elif ip > 0:
                whip = (h + bb) / ip
            else:
                whip = 0.0

            projections.append(
                PitchingProjection(
                    player_id=_get_player_id(row),
                    mlbam_id=_get_mlbam_id(row),
                    name=row.get("name", ""),
                    team=row.get("team", ""),
                    g=_get_int(row, "g"),
                    gs=_get_int(row, "gs"),
                    ip=ip,
                    w=_get_int(row, "w"),
                    l=_get_int(row, "l"),
                    sv=_get_int(row, "sv"),
                    hld=_get_int(row, "hld"),
                    so=_get_int(row, "so"),
                    bb=bb,
                    hbp=_get_int(row, "hbp"),
                    h=h,
                    er=er,
                    hr=_get_int(row, "hr"),
                    era=era,
                    whip=whip,
                    fip=_get_float(row, "fip"),
                    war=_get_float(row, "war"),
                )
            )
        return projections
