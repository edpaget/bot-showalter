"""Shared MLB Stats API helpers."""

import logging

import httpx

logger = logging.getLogger(__name__)

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def fetch_mlb_active_teams(season: int) -> dict[int, str]:
    """Fetch mlbam_id -> team abbreviation for all active MLB players from the MLB API."""
    with httpx.Client(timeout=30) as client:
        teams_resp = client.get(f"{_MLB_API_BASE}/teams", params={"sportId": 1, "season": season})
        teams_resp.raise_for_status()
        team_map: dict[int, str] = {}
        for t in teams_resp.json().get("teams", []):
            team_map[t["id"]] = t["abbreviation"]

        players_resp = client.get(f"{_MLB_API_BASE}/sports/1/players", params={"season": season})
        players_resp.raise_for_status()
        result: dict[int, str] = {}
        for p in players_resp.json().get("people", []):
            team_id = p.get("currentTeam", {}).get("id")
            abbrev = team_map.get(team_id, "") if team_id else ""
            if abbrev:
                result[p["id"]] = abbrev
    logger.debug("Fetched %d active player-team mappings from MLB API", len(result))
    return result
