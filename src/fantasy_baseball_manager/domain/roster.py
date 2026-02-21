import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class RosterEntry:
    player_id: int | None
    yahoo_player_key: str
    player_name: str
    position: str
    roster_status: str
    acquisition_type: str


@dataclass(frozen=True)
class Roster:
    team_key: str
    league_key: str
    season: int
    week: int
    as_of: datetime.date
    entries: tuple[RosterEntry, ...]
    id: int | None = None
