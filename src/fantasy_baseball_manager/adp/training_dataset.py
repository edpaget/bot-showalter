"""Training dataset builder for ML valuation models.

Joins FanGraphs projection CSVs with FantasyPros ADP data by
normalized player name to produce labeled training rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.adp.name_utils import normalize_name

if TYPE_CHECKING:
    from fantasy_baseball_manager.adp.csv_resolver import ADPCSVResolver
    from fantasy_baseball_manager.adp.models import ADPEntry
    from fantasy_baseball_manager.projections.csv_resolver import CSVProjectionResolver
    from fantasy_baseball_manager.projections.models import (
        BattingProjection,
        PitchingProjection,
        ProjectionSystem,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatterTrainingRow:
    """A single batter training example: projection stats + ADP target."""

    player_id: str
    name: str
    team: str
    year: int
    position: str
    pa: int
    hr: int
    r: int
    rbi: int
    sb: int
    bb: int
    so: int
    obp: float
    slg: float
    woba: float
    war: float
    adp: float


@dataclass(frozen=True)
class PitcherTrainingRow:
    """A single pitcher training example: projection stats + ADP target."""

    player_id: str
    name: str
    team: str
    year: int
    ip: float
    w: int
    sv: int
    hld: int
    so: int
    bb: int
    h: int
    er: int
    hr: int
    era: float
    whip: float
    fip: float
    war: float
    adp: float


@dataclass(frozen=True)
class JoinResult:
    """Result of joining projection data with ADP entries for one year."""

    batter_rows: list[BatterTrainingRow]
    pitcher_rows: list[PitcherTrainingRow]
    unmatched_adp: list[str]
    unmatched_batting: list[str]
    unmatched_pitching: list[str]


def build_training_dataset(
    batting: list[BattingProjection],
    pitching: list[PitchingProjection],
    adp_entries: list[ADPEntry],
    year: int,
    *,
    min_pa: int = 50,
    min_ip: float = 20.0,
) -> JoinResult:
    """Join projection data with ADP entries to build training rows.

    Args:
        batting: Batting projections for the year.
        pitching: Pitching projections for the year.
        adp_entries: ADP entries from FantasyPros for the year.
        year: The season year.
        min_pa: Minimum plate appearances to include a batter.
        min_ip: Minimum innings pitched to include a pitcher.

    Returns:
        JoinResult with matched rows and unmatched diagnostics.
    """
    # Build ADP lookup keyed by normalized name
    adp_lookup: dict[str, ADPEntry] = {}
    for entry in adp_entries:
        adp_lookup[normalize_name(entry.name)] = entry

    matched_adp_keys: set[str] = set()
    batter_rows: list[BatterTrainingRow] = []
    pitcher_rows: list[PitcherTrainingRow] = []
    unmatched_batting: list[str] = []
    unmatched_pitching: list[str] = []

    # Match batters
    for bp in batting:
        if bp.pa < min_pa:
            continue
        key = normalize_name(bp.name)
        adp_entry = adp_lookup.get(key)
        if adp_entry is None:
            unmatched_batting.append(bp.name)
            continue

        matched_adp_keys.add(key)
        batter_rows.append(
            BatterTrainingRow(
                player_id=bp.player_id,
                name=bp.name,
                team=bp.team,
                year=year,
                position=bp.position,
                pa=bp.pa,
                hr=bp.hr,
                r=bp.r,
                rbi=bp.rbi,
                sb=bp.sb,
                bb=bp.bb,
                so=bp.so,
                obp=bp.obp,
                slg=bp.slg,
                woba=bp.woba,
                war=bp.war,
                adp=adp_entry.adp,
            )
        )

    # Match pitchers
    for pp in pitching:
        if pp.ip < min_ip:
            continue
        key = normalize_name(pp.name)
        adp_entry = adp_lookup.get(key)
        if adp_entry is None:
            unmatched_pitching.append(pp.name)
            continue

        matched_adp_keys.add(key)
        pitcher_rows.append(
            PitcherTrainingRow(
                player_id=pp.player_id,
                name=pp.name,
                team=pp.team,
                year=year,
                ip=pp.ip,
                w=pp.w,
                sv=pp.sv,
                hld=pp.hld,
                so=pp.so,
                bb=pp.bb,
                h=pp.h,
                er=pp.er,
                hr=pp.hr,
                era=pp.era,
                whip=pp.whip,
                fip=pp.fip,
                war=pp.war,
                adp=adp_entry.adp,
            )
        )

    # Track unmatched ADP entries
    unmatched_adp = [
        entry.name
        for entry in adp_entries
        if normalize_name(entry.name) not in matched_adp_keys
    ]

    return JoinResult(
        batter_rows=batter_rows,
        pitcher_rows=pitcher_rows,
        unmatched_adp=unmatched_adp,
        unmatched_batting=unmatched_batting,
        unmatched_pitching=unmatched_pitching,
    )


def build_multi_year_dataset(
    years: list[int],
    projection_resolver: CSVProjectionResolver,
    adp_resolver: ADPCSVResolver,
    system: ProjectionSystem,
    *,
    min_pa: int = 50,
    min_ip: float = 20.0,
) -> tuple[list[BatterTrainingRow], list[PitcherTrainingRow], dict[int, JoinResult]]:
    """Build training datasets across multiple years.

    Args:
        years: List of years to process.
        projection_resolver: Resolver for projection CSV files.
        adp_resolver: Resolver for ADP CSV files.
        system: Projection system to use.
        min_pa: Minimum plate appearances to include a batter.
        min_ip: Minimum innings pitched to include a pitcher.

    Returns:
        Tuple of (all_batter_rows, all_pitcher_rows, per_year_results).
    """
    from fantasy_baseball_manager.adp.fantasypros_source import FantasyProsADPParser
    from fantasy_baseball_manager.projections.csv_source import CSVProjectionSource

    parser = FantasyProsADPParser()
    all_batters: list[BatterTrainingRow] = []
    all_pitchers: list[PitcherTrainingRow] = []
    per_year: dict[int, JoinResult] = {}

    for year in years:
        try:
            batting_path, pitching_path = projection_resolver.resolve(system, year)
            adp_path = adp_resolver.resolve(year)
        except FileNotFoundError:
            logger.warning("Missing files for year %d, skipping", year)
            continue

        csv_source = CSVProjectionSource(batting_path, pitching_path, system)
        proj_data = csv_source.fetch_projections()
        adp_entries = parser.parse(adp_path)

        result = build_training_dataset(
            list(proj_data.batting),
            list(proj_data.pitching),
            adp_entries,
            year,
            min_pa=min_pa,
            min_ip=min_ip,
        )

        per_year[year] = result
        all_batters.extend(result.batter_rows)
        all_pitchers.extend(result.pitcher_rows)

    return all_batters, all_pitchers, per_year
