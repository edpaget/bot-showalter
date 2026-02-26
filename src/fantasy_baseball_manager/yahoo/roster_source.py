import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain.roster import Roster, RosterEntry

if TYPE_CHECKING:
    import datetime

    from fantasy_baseball_manager.yahoo.client import YahooFantasyClient
    from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    "IL": "il",
    "IL+": "il",
    "DL": "il",
    "BN": "bench",
    "NA": "na",
}


class YahooRosterSource:
    def __init__(self, client: YahooFantasyClient, mapper: YahooPlayerMapper) -> None:
        self._client = client
        self._mapper = mapper

    def fetch_team_roster(
        self,
        *,
        team_key: str,
        league_key: str,
        season: int,
        week: int,
        as_of: datetime.date,
    ) -> Roster:
        data = self._client.get_roster(team_key)
        entries = self._parse_roster_entries(data)
        return Roster(
            team_key=team_key,
            league_key=league_key,
            season=season,
            week=week,
            as_of=as_of,
            entries=tuple(entries),
        )

    def fetch_all_rosters(
        self,
        *,
        team_keys: list[str],
        league_key: str,
        season: int,
        week: int,
        as_of: datetime.date,
    ) -> list[Roster]:
        rosters: list[Roster] = []
        for team_key in team_keys:
            roster = self.fetch_team_roster(
                team_key=team_key,
                league_key=league_key,
                season=season,
                week=week,
                as_of=as_of,
            )
            rosters.append(roster)
        return rosters

    def _parse_roster_entries(self, data: dict[str, Any]) -> list[RosterEntry]:
        entries: list[RosterEntry] = []
        team_section = data["fantasy_content"]["team"]
        roster_data = team_section[1]["roster"]["0"]["players"]

        for key, value in roster_data.items():
            if key == "count":
                continue

            player_info = value["player"]
            player_meta = player_info[0]

            # Extract player data from the list of dicts
            player_data = self._extract_player_data(player_meta)

            # Extract selected position
            selected_position = self._extract_selected_position(player_info)

            # Extract transaction/acquisition type
            acquisition_type = self._extract_acquisition_type(player_info)

            # Resolve player mapping
            mapping = self._mapper.resolve(player_data)
            player_id = mapping.player_id if mapping is not None else None

            roster_status = _infer_roster_status(selected_position)

            entries.append(
                RosterEntry(
                    player_id=player_id,
                    yahoo_player_key=player_data["player_key"],
                    player_name=player_data["name"],
                    position=selected_position,
                    roster_status=roster_status,
                    acquisition_type=acquisition_type,
                )
            )

        return entries

    @staticmethod
    def _extract_player_data(player_meta: list[Any]) -> dict[str, Any]:
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
                    result["eligible_positions"] = [
                        p["position"] for p in item["eligible_positions"] if isinstance(p, dict)
                    ]
                elif "player_id" in item:
                    result["player_id"] = item["player_id"]
        return result

    @staticmethod
    def _extract_selected_position(player_info: list[Any]) -> str:
        for item in player_info:
            if isinstance(item, dict) and "selected_position" in item:
                pos_list = item["selected_position"]
                for pos_item in pos_list:
                    if isinstance(pos_item, dict) and "position" in pos_item:
                        return pos_item["position"]
        return ""

    @staticmethod
    def _extract_acquisition_type(player_info: list[Any]) -> str:
        for item in player_info:
            if isinstance(item, dict) and "transaction_data" in item:
                return item["transaction_data"].get("type", "")
        return ""


def _infer_roster_status(selected_position: str) -> str:
    return _STATUS_MAP.get(selected_position, "active")
