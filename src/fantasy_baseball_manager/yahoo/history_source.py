from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import SeasonData

if TYPE_CHECKING:
    import datetime

    from fantasy_baseball_manager.yahoo.client import YahooFantasyClient
    from fantasy_baseball_manager.yahoo.draft_source import YahooDraftSource
    from fantasy_baseball_manager.yahoo.roster_source import YahooRosterSource


class YahooLeagueHistorySource:
    def __init__(
        self,
        client: YahooFantasyClient,
        draft_source: YahooDraftSource,
        roster_source: YahooRosterSource,
    ) -> None:
        self._client = client
        self._draft_source = draft_source
        self._roster_source = roster_source

    def discover_seasons(self, league_id: int) -> list[tuple[str, int]]:
        seasons = self._client.get_available_seasons()
        return [(f"{game_key}.l.{league_id}", season) for game_key, season in seasons]

    def fetch_season_data(
        self,
        league_key: str,
        season: int,
        team_keys: list[str],
        as_of: datetime.date,
    ) -> SeasonData:
        draft_picks = self._draft_source.fetch_draft_results(league_key, season)
        rosters = self._roster_source.fetch_all_rosters(
            team_keys=team_keys,
            league_key=league_key,
            season=season,
            week=1,
            as_of=as_of,
        )
        return SeasonData(
            league_key=league_key,
            season=season,
            draft_picks=draft_picks,
            rosters=rosters,
        )
