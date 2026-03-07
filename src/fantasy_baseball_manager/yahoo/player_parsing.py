import contextlib
from typing import Any

from fantasy_baseball_manager.domain import position_from_raw


def extract_player_data(player_meta: list[Any]) -> dict[str, Any]:
    """Extract player fields from Yahoo's nested player metadata list."""
    result: dict[str, Any] = {}
    for item in player_meta:
        if isinstance(item, dict):
            if "player_key" in item:
                result["player_key"] = item["player_key"]
            elif "name" in item:
                result["name"] = item["name"]["full"]
            elif "editorial_team_abbr" in item:
                result["editorial_team_abbr"] = item["editorial_team_abbr"]
            elif "eligible_positions" in item:
                result["eligible_positions"] = _normalize_positions(
                    [p["position"] for p in item["eligible_positions"] if isinstance(p, dict)]
                )
            elif "player_id" in item:
                result["player_id"] = item["player_id"]
    return result


def _normalize_positions(raw_positions: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw in raw_positions:
        with contextlib.suppress(ValueError):
            normalized.append(position_from_raw(raw))
    return normalized
