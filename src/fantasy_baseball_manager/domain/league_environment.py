from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueEnvironment:
    league: str
    season: int
    level: str
    runs_per_game: float
    avg: float
    obp: float
    slg: float
    k_pct: float
    bb_pct: float
    hr_per_pa: float
    babip: float
    id: int | None = None
    loaded_at: str | None = None
