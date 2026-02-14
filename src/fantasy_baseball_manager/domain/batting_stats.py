from dataclasses import dataclass


@dataclass(frozen=True)
class BattingStats:
    player_id: int
    season: int
    source: str
    id: int | None = None
    team_id: int | None = None
    pa: int | None = None
    ab: int | None = None
    h: int | None = None
    doubles: int | None = None
    triples: int | None = None
    hr: int | None = None
    rbi: int | None = None
    r: int | None = None
    sb: int | None = None
    cs: int | None = None
    bb: int | None = None
    so: int | None = None
    hbp: int | None = None
    sf: int | None = None
    sh: int | None = None
    gdp: int | None = None
    ibb: int | None = None
    avg: float | None = None
    obp: float | None = None
    slg: float | None = None
    ops: float | None = None
    woba: float | None = None
    wrc_plus: float | None = None
    war: float | None = None
    loaded_at: str | None = None
