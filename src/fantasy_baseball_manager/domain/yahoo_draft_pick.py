from dataclasses import dataclass


@dataclass(frozen=True)
class YahooDraftPick:
    league_key: str
    season: int
    round: int
    pick: int
    team_key: str
    yahoo_player_key: str
    player_id: int | None
    player_name: str
    position: str
    cost: int | None = None
    id: int | None = None
