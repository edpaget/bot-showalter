from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.player.identity import Player

if TYPE_CHECKING:
    from fantasy_baseball_manager.league.roster import RosterSource
    from fantasy_baseball_manager.player_id.mapper import SfbbMapper


@dataclass(frozen=True)
class TeamKeeperInfo:
    team_key: str
    team_name: str
    candidate_ids: tuple[str, ...]
    candidate_position_types: tuple[str, ...]
    candidate_positions: dict[tuple[str, str], tuple[str, ...]]


@dataclass(frozen=True)
class LeagueKeeperData:
    teams: tuple[TeamKeeperInfo, ...]
    unmapped_yahoo_ids: tuple[str, ...]


@dataclass(frozen=True)
class YahooKeeperData:
    user_candidate_ids: tuple[str, ...]
    user_candidate_position_types: tuple[str, ...]
    user_candidate_positions: dict[tuple[str, str], tuple[str, ...]]
    other_keeper_ids: frozenset[str]
    unmapped_yahoo_ids: tuple[str, ...]


class YahooKeeperSource:
    """Transforms Yahoo roster data into keeper calculation inputs."""

    def __init__(
        self,
        roster_source: RosterSource,
        id_mapper: SfbbMapper,
        user_team_key: str,
    ) -> None:
        self._roster_source = roster_source
        self._id_mapper = id_mapper
        self._user_team_key = user_team_key

    def fetch_keeper_data(self) -> YahooKeeperData:
        rosters = self._roster_source.fetch_rosters()

        user_team = None
        other_teams = []
        for team in rosters.teams:
            if team.team_key == self._user_team_key:
                user_team = team
            else:
                other_teams.append(team)

        if user_team is None:
            msg = f"User team key {self._user_team_key!r} not found in league rosters"
            raise ValueError(msg)

        unmapped: list[str] = []
        candidate_ids: list[str] = []
        candidate_position_types: list[str] = []
        candidate_positions: dict[tuple[str, str], tuple[str, ...]] = {}

        for roster_player in user_team.players:
            enriched = self._id_mapper(Player(name=roster_player.name, yahoo_id=roster_player.yahoo_id)).unwrap()
            if enriched.fangraphs_id is None:
                unmapped.append(roster_player.yahoo_id)
            else:
                candidate_ids.append(enriched.fangraphs_id)
                candidate_position_types.append(roster_player.position_type)
                candidate_positions[(enriched.fangraphs_id, roster_player.position_type)] = (
                    roster_player.eligible_positions
                )

        other_keeper_ids: set[str] = set()
        for team in other_teams:
            for roster_player in team.players:
                enriched = self._id_mapper(Player(name=roster_player.name, yahoo_id=roster_player.yahoo_id)).unwrap()
                if enriched.fangraphs_id is None:
                    unmapped.append(roster_player.yahoo_id)
                else:
                    other_keeper_ids.add(enriched.fangraphs_id)

        return YahooKeeperData(
            user_candidate_ids=tuple(candidate_ids),
            user_candidate_position_types=tuple(candidate_position_types),
            user_candidate_positions=candidate_positions,
            other_keeper_ids=frozenset(other_keeper_ids),
            unmapped_yahoo_ids=tuple(unmapped),
        )

    def fetch_league_keeper_data(self) -> LeagueKeeperData:
        """Fetch keeper candidate data for all teams in the league."""
        rosters = self._roster_source.fetch_rosters()

        unmapped: list[str] = []
        team_infos: list[TeamKeeperInfo] = []

        for team in rosters.teams:
            candidate_ids: list[str] = []
            candidate_position_types: list[str] = []
            candidate_positions: dict[tuple[str, str], tuple[str, ...]] = {}

            for roster_player in team.players:
                enriched = self._id_mapper(Player(name=roster_player.name, yahoo_id=roster_player.yahoo_id)).unwrap()
                if enriched.fangraphs_id is None:
                    unmapped.append(roster_player.yahoo_id)
                else:
                    candidate_ids.append(enriched.fangraphs_id)
                    candidate_position_types.append(roster_player.position_type)
                    candidate_positions[(enriched.fangraphs_id, roster_player.position_type)] = (
                        roster_player.eligible_positions
                    )

            team_infos.append(
                TeamKeeperInfo(
                    team_key=team.team_key,
                    team_name=team.team_name,
                    candidate_ids=tuple(candidate_ids),
                    candidate_position_types=tuple(candidate_position_types),
                    candidate_positions=candidate_positions,
                )
            )

        return LeagueKeeperData(
            teams=tuple(team_infos),
            unmapped_yahoo_ids=tuple(unmapped),
        )
