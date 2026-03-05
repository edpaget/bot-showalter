import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import Roster, RosterEntry
from fantasy_baseball_manager.yahoo.player_parsing import extract_player_data

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
        week: int | None = None,
        as_of: datetime.date,
    ) -> Roster:
        data = self._client.get_roster(team_key, week=week)
        entries = self._parse_roster_entries(data)
        logger.debug("Fetched roster for %s: %d entries (week=%s)", team_key, len(entries), week)
        return Roster(
            team_key=team_key,
            league_key=league_key,
            season=season,
            week=week if week is not None else 0,
            as_of=as_of,
            entries=tuple(entries),
        )

    def fetch_all_rosters(  # pragma: no cover
        self,
        *,
        team_keys: list[str],
        league_key: str,
        season: int,
        week: int | None = None,
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
        roster_section = team_section[1].get("roster")
        if roster_section is None:
            return entries
        roster_data = roster_section["0"]["players"]

        # Yahoo returns an empty list when there are no players, but a dict otherwise.
        if isinstance(roster_data, list):
            return entries

        for key, value in roster_data.items():
            if key == "count":
                continue

            player_info = value["player"]
            player_meta = player_info[0]

            # Extract player data from the list of dicts
            player_data = extract_player_data(player_meta)

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
    def _extract_selected_position(player_info: list[Any]) -> str:  # pragma: no cover
        for item in player_info:
            if isinstance(item, dict) and "selected_position" in item:
                pos_list = item["selected_position"]
                for pos_item in pos_list:
                    if isinstance(pos_item, dict) and "position" in pos_item:
                        return pos_item["position"]
        return ""

    @staticmethod
    def _extract_acquisition_type(player_info: list[Any]) -> str:  # pragma: no cover
        for item in player_info:
            if isinstance(item, dict) and "transaction_data" in item:
                return item["transaction_data"].get("type", "")
        return ""


def _infer_roster_status(selected_position: str) -> str:
    return _STATUS_MAP.get(selected_position, "active")
