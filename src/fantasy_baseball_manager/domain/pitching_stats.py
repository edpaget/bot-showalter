from dataclasses import dataclass


@dataclass(frozen=True)
class PitchingStats:
    player_id: int
    season: int
    source: str
    id: int | None = None
    team_id: int | None = None
    w: int | None = None
    l: int | None = None  # noqa: E741
    era: float | None = None
    g: int | None = None
    gs: int | None = None
    sv: int | None = None
    hld: int | None = None
    ip: float | None = None
    h: int | None = None
    er: int | None = None
    hr: int | None = None
    bb: int | None = None
    so: int | None = None
    whip: float | None = None
    k_per_9: float | None = None
    bb_per_9: float | None = None
    fip: float | None = None
    xfip: float | None = None
    war: float | None = None
    loaded_at: str | None = None
