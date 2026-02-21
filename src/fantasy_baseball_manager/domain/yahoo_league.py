from dataclasses import dataclass


@dataclass(frozen=True)
class YahooLeague:
    league_key: str
    name: str
    season: int
    num_teams: int
    draft_type: str
    is_keeper: bool
    game_key: str
    id: int | None = None


@dataclass(frozen=True)
class YahooTeam:
    team_key: str
    league_key: str
    team_id: int
    name: str
    manager_name: str
    is_owned_by_user: bool
    id: int | None = None
