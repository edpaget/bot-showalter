"""Data models for external projection systems."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ProjectionSystem(Enum):
    """Supported projection systems."""

    STEAMER = "steamer"
    ZIPS = "zips"
    ZIPS_DC = "zipsdc"  # ZiPS with depth chart playing time


@dataclass(frozen=True)
class BattingProjection:
    """A single player's batting projection from an external system.

    Attributes:
        player_id: FanGraphs player ID.
        mlbam_id: MLB Advanced Media ID for cross-referencing.
        name: Player's display name.
        team: Team abbreviation.
        position: Primary position.
        g: Projected games.
        pa: Projected plate appearances.
        ab: Projected at-bats.
        h: Projected hits.
        singles: Projected singles (1B).
        doubles: Projected doubles (2B).
        triples: Projected triples (3B).
        hr: Projected home runs.
        r: Projected runs.
        rbi: Projected RBIs.
        sb: Projected stolen bases.
        cs: Projected caught stealing.
        bb: Projected walks.
        so: Projected strikeouts.
        hbp: Projected hit by pitches.
        sf: Projected sacrifice flies.
        sh: Projected sacrifice hits.
        obp: Projected on-base percentage.
        slg: Projected slugging percentage.
        ops: Projected OPS.
        woba: Projected wOBA.
        war: Projected WAR.
    """

    player_id: str
    mlbam_id: str | None
    name: str
    team: str
    position: str
    g: int
    pa: int
    ab: int
    h: int
    singles: int
    doubles: int
    triples: int
    hr: int
    r: int
    rbi: int
    sb: int
    cs: int
    bb: int
    so: int
    hbp: int
    sf: int
    sh: int
    obp: float
    slg: float
    ops: float
    woba: float
    war: float


@dataclass(frozen=True)
class PitchingProjection:
    """A single player's pitching projection from an external system.

    Attributes:
        player_id: FanGraphs player ID.
        mlbam_id: MLB Advanced Media ID for cross-referencing.
        name: Player's display name.
        team: Team abbreviation.
        g: Projected games.
        gs: Projected games started.
        ip: Projected innings pitched.
        w: Projected wins.
        l: Projected losses.
        sv: Projected saves.
        hld: Projected holds.
        so: Projected strikeouts.
        bb: Projected walks.
        hbp: Projected hit by pitches.
        h: Projected hits allowed.
        er: Projected earned runs.
        hr: Projected home runs allowed.
        era: Projected ERA.
        whip: Projected WHIP.
        fip: Projected FIP.
        war: Projected WAR.
    """

    player_id: str
    mlbam_id: str | None
    name: str
    team: str
    g: int
    gs: int
    ip: float
    w: int
    l: int
    sv: int
    hld: int
    so: int
    bb: int
    hbp: int
    h: int
    er: int
    hr: int
    era: float
    whip: float
    fip: float
    war: float


@dataclass(frozen=True)
class ProjectionData:
    """Collection of projections from an external system.

    Attributes:
        batting: Tuple of batting projections.
        pitching: Tuple of pitching projections.
        system: The projection system (steamer, zips, etc.).
        fetched_at: When this data was fetched.
    """

    batting: tuple[BattingProjection, ...]
    pitching: tuple[PitchingProjection, ...]
    system: ProjectionSystem
    fetched_at: datetime
