import logging
from typing import Any

from fantasy_baseball_manager.yahoo.client import YahooFantasyClient

logger = logging.getLogger(__name__)


class YahooLeagueSource:
    def __init__(self, client: YahooFantasyClient) -> None:
        self._client = client

    @property
    def source_type(self) -> str:
        return "yahoo_league"

    def fetch(self, *, league_key: str, game_key: str) -> dict[str, Any]:
        """Fetch league settings and teams from Yahoo API.

        Returns a dict with 'league' and 'teams' keys ready for repo upsert.
        """
        settings_data = self._client.get_league_settings(league_key)
        teams_data = self._client.get_teams(league_key)

        league_info = self._parse_league(settings_data, game_key)
        teams = self._parse_teams(teams_data, league_key)

        return {"league": league_info, "teams": teams}

    @staticmethod
    def _parse_league(data: dict[str, Any], game_key: str) -> dict[str, Any]:
        league_parts = data["fantasy_content"]["league"]
        meta = league_parts[0]
        settings = league_parts[1]["settings"][0]

        is_keeper = settings.get("uses_keeper", "0") == "1"

        return {
            "league_key": meta["league_key"],
            "name": meta["name"],
            "season": int(meta["season"]),
            "num_teams": meta["num_teams"],
            "draft_type": settings["draft_type"],
            "is_keeper": is_keeper,
            "game_key": game_key,
        }

    @staticmethod
    def _parse_teams(data: dict[str, Any], league_key: str) -> list[dict[str, Any]]:
        teams_section = data["fantasy_content"]["league"][1]["teams"]
        teams: list[dict[str, Any]] = []

        for key, value in teams_section.items():
            if key == "count":
                continue
            team_info = value["team"][0]

            # Yahoo's team structure: list of dicts with various fields
            team_key = ""
            team_id = 0
            name = ""
            manager_name = ""
            is_owned_by_user = False

            for item in team_info:
                if isinstance(item, dict):
                    if "team_key" in item:
                        team_key = item["team_key"]
                    elif "team_id" in item:
                        team_id = int(item["team_id"])
                    elif "name" in item:
                        name = item["name"]
                    elif "managers" in item:
                        mgr = item["managers"][0]["manager"]
                        manager_name = mgr.get("nickname", "")
                        is_owned_by_user = mgr.get("is_current_login", "0") == "1"

            teams.append(
                {
                    "team_key": team_key,
                    "league_key": league_key,
                    "team_id": team_id,
                    "name": name,
                    "manager_name": manager_name,
                    "is_owned_by_user": is_owned_by_user,
                }
            )

        return teams
