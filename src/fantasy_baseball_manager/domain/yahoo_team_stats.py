from dataclasses import dataclass


@dataclass(frozen=True)
class TeamSeasonStats:
    team_key: str
    league_key: str
    season: int
    team_name: str
    final_rank: int
    stat_values: dict[str, float]
    id: int | None = None
