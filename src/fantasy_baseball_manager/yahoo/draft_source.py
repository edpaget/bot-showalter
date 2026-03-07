import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import YahooDraftPick
from fantasy_baseball_manager.yahoo.player_parsing import extract_player_data

if TYPE_CHECKING:
    from fantasy_baseball_manager.yahoo.client import YahooFantasyClient
    from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper

logger = logging.getLogger(__name__)


class YahooDraftSource:
    def __init__(self, client: YahooFantasyClient, mapper: YahooPlayerMapper) -> None:
        self._client = client
        self._mapper = mapper

    def fetch_draft_results(self, league_key: str, season: int) -> list[YahooDraftPick]:
        data = self._client.get_draft_results(league_key)
        raw_picks = self._parse_raw_picks(data)
        if not raw_picks:
            return []

        # Batch-fetch player details for names and positions
        player_keys = [p["player_key"] for p in raw_picks]
        player_details = self._fetch_player_details(league_key, player_keys)

        picks: list[YahooDraftPick] = []
        for raw in raw_picks:
            player_key = raw["player_key"]
            details = player_details.get(player_key, {})

            name = details.get("name", "Unknown")
            position = self._primary_position(details.get("eligible_positions", []))

            # Resolve player ID via mapper
            mapping = self._mapper.resolve(details) if details else None
            player_id = mapping.player_id if mapping is not None else None

            cost_str = raw.get("cost")
            cost = int(cost_str) if cost_str is not None else None

            picks.append(
                YahooDraftPick(
                    league_key=league_key,
                    season=season,
                    round=int(raw["round"]),
                    pick=int(raw["pick"]),
                    team_key=raw["team_key"],
                    yahoo_player_key=player_key,
                    player_id=player_id,
                    player_name=name,
                    position=position,
                    cost=cost,
                )
            )

        return picks

    @staticmethod
    def _parse_raw_picks(data: dict[str, Any]) -> list[dict[str, Any]]:
        league_data = data.get("fantasy_content", {}).get("league")
        if not isinstance(league_data, list) or len(league_data) < 2:
            logger.warning("Unexpected draft results response structure")
            return []

        draft_results = league_data[1].get("draft_results")
        if not isinstance(draft_results, dict):
            logger.warning("No draft_results found in response")
            return []

        raw_picks: list[dict[str, Any]] = []
        for key, value in draft_results.items():
            if key == "count":
                continue
            raw_picks.append(value["draft_result"])
        return raw_picks

    def _fetch_player_details(self, league_key: str, player_keys: list[str]) -> dict[str, dict[str, Any]]:
        if not player_keys:
            return {}

        # Yahoo API limits batch size; fetch in chunks of 25
        details: dict[str, dict[str, Any]] = {}
        for i in range(0, len(player_keys), 25):
            chunk = player_keys[i : i + 25]
            data = self._client.get_players(league_key, chunk)
            parsed = self._parse_players_response(data)
            details.update(parsed)
        return details

    @staticmethod
    def _parse_players_response(
        data: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        league_section = data.get("fantasy_content", {}).get("league")
        if not isinstance(league_section, list) or len(league_section) < 2:
            return result

        players_section = league_section[1].get("players", {})
        for key, value in players_section.items():
            if key == "count":
                continue
            player_info = value["player"]
            player_meta = player_info[0]
            player_data = extract_player_data(player_meta)
            if "player_key" in player_data:
                result[player_data["player_key"]] = player_data
        return result

    @staticmethod
    def _primary_position(positions: list[str]) -> str:
        non_generic = [p for p in positions if p not in ("UTIL", "BN", "IL", "IL+", "NA", "DL")]
        return non_generic[0] if non_generic else (positions[0] if positions else "UTIL")
