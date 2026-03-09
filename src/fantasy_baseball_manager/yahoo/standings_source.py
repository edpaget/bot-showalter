import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import TeamSeasonStats

if TYPE_CHECKING:
    from fantasy_baseball_manager.yahoo.client import YahooFantasyClient

logger = logging.getLogger(__name__)

# Yahoo stat_id → fbm category key.  Unmapped IDs are silently ignored.
STAT_ID_MAP: dict[str, str] = {
    "4": "obp",
    "7": "r",
    "12": "hr",
    "13": "rbi",
    "16": "sb",
    "26": "era",
    "27": "whip",
    "28": "w",
    "42": "so",
    "50": "ip",
    "90": "sv+hld",
}


class YahooStandingsSource:
    def __init__(self, client: YahooFantasyClient) -> None:
        self._client = client

    def fetch(self, league_key: str, season: int) -> list[TeamSeasonStats]:
        data = self._client.get_standings(league_key)
        return self._parse_standings(data, league_key, season)

    @classmethod
    def _parse_standings(cls, data: dict[str, Any], league_key: str, season: int) -> list[TeamSeasonStats]:
        teams_section = data["fantasy_content"]["league"][1]["standings"][0]["teams"]
        results: list[TeamSeasonStats] = []

        for key, value in teams_section.items():
            if key == "count":
                continue

            team_data = value["team"]
            team_info = team_data[0]
            team_stats = team_data[1]["team_stats"]["stats"]
            team_standings = team_data[2]["team_standings"]

            # Extract team name and key from the info list
            team_key = ""
            team_name = ""
            for item in team_info:
                if isinstance(item, dict):
                    if "team_key" in item:
                        team_key = item["team_key"]
                    elif "name" in item:
                        team_name = item["name"]

            raw_rank = team_standings["rank"]
            final_rank = int(raw_rank) if raw_rank else 0

            # Parse stat values
            stat_values: dict[str, float] = {}
            for stat_entry in team_stats:
                stat = stat_entry["stat"]
                stat_id = stat["stat_id"]
                raw_value = stat["value"]
                if stat_id not in STAT_ID_MAP or raw_value == "":
                    continue
                stat_values[STAT_ID_MAP[stat_id]] = float(raw_value)

            results.append(
                TeamSeasonStats(
                    team_key=team_key,
                    league_key=league_key,
                    season=season,
                    team_name=team_name,
                    final_rank=final_rank,
                    stat_values=stat_values,
                )
            )

        return results
