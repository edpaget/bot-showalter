from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.league.roster import RosterSource
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper


@dataclass(frozen=True)
class YahooKeeperData:
    user_candidate_ids: tuple[str, ...]
    user_candidate_positions: dict[str, tuple[str, ...]]
    other_keeper_ids: frozenset[str]
    unmapped_yahoo_ids: tuple[str, ...]


class YahooKeeperSource:
    """Transforms Yahoo roster data into keeper calculation inputs."""

    def __init__(
        self,
        roster_source: RosterSource,
        id_mapper: PlayerIdMapper,
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
        candidate_positions: dict[str, tuple[str, ...]] = {}

        for player in user_team.players:
            fg_id = self._id_mapper.yahoo_to_fangraphs(player.yahoo_id)
            if fg_id is None:
                unmapped.append(player.yahoo_id)
            else:
                candidate_ids.append(fg_id)
                candidate_positions[fg_id] = player.eligible_positions

        other_keeper_ids: set[str] = set()
        for team in other_teams:
            for player in team.players:
                fg_id = self._id_mapper.yahoo_to_fangraphs(player.yahoo_id)
                if fg_id is None:
                    unmapped.append(player.yahoo_id)
                else:
                    other_keeper_ids.add(fg_id)

        return YahooKeeperData(
            user_candidate_ids=tuple(candidate_ids),
            user_candidate_positions=candidate_positions,
            other_keeper_ids=frozenset(other_keeper_ids),
            unmapped_yahoo_ids=tuple(unmapped),
        )
