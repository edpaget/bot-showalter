from dataclasses import dataclass, field


@dataclass(frozen=True)
class YahooDraftSetupInfo:
    num_teams: int
    draft_format: str
    user_team_id: int
    team_names: dict[int, str]
    draft_order: list[int]
    is_keeper: bool
    max_keepers: int | None
    keeper_player_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class YahooLeague:
    league_key: str
    name: str
    season: int
    num_teams: int
    draft_type: str
    is_keeper: bool
    game_key: str
    renew: str | None = None
    id: int | None = None


@dataclass(frozen=True)
class YahooLeagueInfo:
    league_key: str
    league_name: str
    season: int
    num_teams: int
    is_keeper: bool
    max_keepers: int | None
    user_team_name: str | None


@dataclass(frozen=True)
class YahooTeam:
    team_key: str
    league_key: str
    team_id: int
    name: str
    manager_name: str
    is_owned_by_user: bool
    id: int | None = None
