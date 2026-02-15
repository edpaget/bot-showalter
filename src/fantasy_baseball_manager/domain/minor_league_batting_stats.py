from dataclasses import dataclass


@dataclass(frozen=True)
class MinorLeagueBattingStats:
    player_id: int
    season: int
    level: str
    league: str
    team: str
    g: int
    pa: int
    ab: int
    h: int
    doubles: int
    triples: int
    hr: int
    r: int
    rbi: int
    bb: int
    so: int
    sb: int
    cs: int
    avg: float
    obp: float
    slg: float
    age: float
    id: int | None = None
    hbp: int | None = None
    sf: int | None = None
    sh: int | None = None
    loaded_at: str | None = None
