from dataclasses import dataclass


@dataclass(frozen=True)
class RosterStint:
    player_id: int
    team_id: int
    season: int
    start_date: str
    id: int | None = None
    end_date: str | None = None
    loaded_at: str | None = None
