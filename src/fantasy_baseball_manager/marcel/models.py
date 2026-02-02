from dataclasses import dataclass


@dataclass(frozen=True)
class BattingSeasonStats:
    player_id: str
    name: str
    year: int
    age: int
    pa: int
    ab: int
    h: int
    singles: int
    doubles: int
    triples: int
    hr: int
    bb: int
    so: int
    hbp: int
    sf: int
    sh: int
    sb: int
    cs: int
    r: int
    rbi: int
    team: str = ""


@dataclass(frozen=True)
class PitchingSeasonStats:
    player_id: str
    name: str
    year: int
    age: int
    ip: float
    g: int
    gs: int
    er: int
    h: int
    bb: int
    so: int
    hr: int
    hbp: int
    w: int
    sv: int
    hld: int
    bs: int
    team: str = ""


@dataclass(frozen=True)
class BattingProjection:
    player_id: str
    name: str
    year: int
    age: int
    pa: float
    ab: float
    h: float
    singles: float
    doubles: float
    triples: float
    hr: float
    bb: float
    so: float
    hbp: float
    sf: float
    sh: float
    sb: float
    cs: float
    r: float
    rbi: float


@dataclass(frozen=True)
class PitchingProjection:
    player_id: str
    name: str
    year: int
    age: int
    ip: float
    g: float
    gs: float
    er: float
    h: float
    bb: float
    so: float
    hr: float
    hbp: float
    era: float
    whip: float
    w: float
    nsvh: float
