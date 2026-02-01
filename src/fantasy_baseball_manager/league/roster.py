from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster

if TYPE_CHECKING:
    import yahoo_fantasy_api


class RosterSource(Protocol):
    def fetch_rosters(self) -> LeagueRosters: ...


class YahooRosterSource:
    """Fetches rosters from a Yahoo Fantasy league."""

    def __init__(self, league: "yahoo_fantasy_api.League") -> None:
        self._league = league

    def fetch_rosters(self) -> LeagueRosters:
        teams_data = self._league.teams()
        team_rosters: list[TeamRoster] = []

        for team_key, team_info in teams_data.items():
            team_name = team_info["name"]
            roster_data = self._league.to_team(team_key).roster()
            players: list[RosterPlayer] = []

            for player in roster_data:
                players.append(
                    RosterPlayer(
                        yahoo_id=str(player["player_id"]),
                        name=player["name"],
                        position_type=player["position_type"],
                        eligible_positions=tuple(player.get("eligible_positions", ())),
                    )
                )

            team_rosters.append(
                TeamRoster(
                    team_key=team_key,
                    team_name=team_name,
                    players=tuple(players),
                )
            )

        return LeagueRosters(
            league_key=self._league.league_id,
            teams=tuple(team_rosters),
        )
