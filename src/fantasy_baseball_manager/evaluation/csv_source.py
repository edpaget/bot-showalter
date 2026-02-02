import csv
from pathlib import Path

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection


def _normalize_headers(reader: csv.DictReader) -> list[dict[str, str]]:
    """Read all rows with case-insensitive header normalization."""
    if reader.fieldnames is None:
        return []
    lower_map = {name: name.strip().lower() for name in reader.fieldnames}
    rows: list[dict[str, str]] = []
    for row in reader:
        normalized = {lower_map[k]: v for k, v in row.items() if k in lower_map}
        rows.append(normalized)
    return rows


class CsvProjectionSource:
    def __init__(
        self,
        batting_path: Path | None,
        pitching_path: Path | None,
    ) -> None:
        self._batting_path = batting_path
        self._pitching_path = pitching_path

    def batting_projections(self) -> list[BattingProjection]:
        if self._batting_path is None:
            return []
        with open(self._batting_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = _normalize_headers(reader)

        projections: list[BattingProjection] = []
        for row in rows:
            h = float(row["h"])
            doubles = float(row["2b"])
            triples = float(row["3b"])
            hr = float(row["hr"])
            singles = float(row["1b"]) if "1b" in row else h - doubles - triples - hr
            projections.append(
                BattingProjection(
                    player_id=row["idfg"],
                    name=row["name"],
                    year=0,
                    age=int(row.get("age", "0")),
                    pa=float(row["pa"]),
                    ab=float(row["ab"]),
                    h=h,
                    singles=singles,
                    doubles=doubles,
                    triples=triples,
                    hr=hr,
                    bb=float(row["bb"]),
                    so=float(row["so"]),
                    hbp=float(row.get("hbp", "0")),
                    sf=float(row.get("sf", "0")),
                    sh=float(row.get("sh", "0")),
                    sb=float(row.get("sb", "0")),
                    cs=float(row.get("cs", "0")),
                    r=float(row.get("r", "0")),
                    rbi=float(row.get("rbi", "0")),
                )
            )
        return projections

    def pitching_projections(self) -> list[PitchingProjection]:
        if self._pitching_path is None:
            return []
        with open(self._pitching_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = _normalize_headers(reader)

        projections: list[PitchingProjection] = []
        for row in rows:
            ip = float(row["ip"])
            er = float(row["er"])
            h = float(row["h"])
            bb = float(row["bb"])
            era = (er / ip * 9) if ip > 0 else 0.0
            whip = ((h + bb) / ip) if ip > 0 else 0.0
            projections.append(
                PitchingProjection(
                    player_id=row["idfg"],
                    name=row["name"],
                    year=0,
                    age=int(row.get("age", "0")),
                    ip=ip,
                    g=float(row.get("g", "0")),
                    gs=float(row.get("gs", "0")),
                    er=er,
                    h=h,
                    bb=bb,
                    so=float(row["so"]),
                    hr=float(row.get("hr", "0")),
                    hbp=float(row.get("hbp", "0")),
                    era=era,
                    whip=whip,
                )
            )
        return projections
