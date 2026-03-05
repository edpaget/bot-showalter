"""Player-to-team mapping provider backed by Lahman stints + MLB API overlay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fantasy_baseball_manager.team_aliases import to_modern

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.repos import PlayerRepo, RosterStintRepo, TeamRepo

logger = logging.getLogger(__name__)


class MlbApiPlayerTeamProvider:
    """Builds player_id -> modern team abbreviation mappings.

    Combines Lahman roster stints as a baseline with an MLB API overlay
    for current-season accuracy. Caches results per season.
    """

    def __init__(
        self,
        player_repo: PlayerRepo,
        team_repo: TeamRepo,
        roster_stint_repo: RosterStintRepo,
        fetcher: Callable[[int], dict[int, str]] | None = None,
    ) -> None:
        self._player_repo = player_repo
        self._team_repo = team_repo
        self._roster_stint_repo = roster_stint_repo
        self._fetcher = fetcher
        self._cache: dict[int, dict[int, str]] = {}

    def get_player_teams(self, season: int) -> dict[int, str]:
        if season in self._cache:
            return self._cache[season]

        # Lahman baseline
        teams = {t.id: t.abbreviation for t in self._team_repo.all() if t.id is not None}
        stints = self._roster_stint_repo.get_by_season(season)
        if not stints:
            stints = self._roster_stint_repo.get_by_season(season - 1)

        player_teams: dict[int, str] = {}
        for stint in stints:
            abbrev = teams.get(stint.team_id)
            if abbrev:
                player_teams[stint.player_id] = to_modern(abbrev)

        # MLB API overlay
        if self._fetcher is not None:
            try:
                mlbam_teams = self._fetcher(season)
                mlbam_to_id = {
                    p.mlbam_id: p.id for p in self._player_repo.all() if p.mlbam_id is not None and p.id is not None
                }
                updated = 0
                for mlbam_id, abbrev in mlbam_teams.items():
                    pid = mlbam_to_id.get(mlbam_id)
                    if pid is not None and player_teams.get(pid) != abbrev:
                        player_teams[pid] = abbrev
                        updated += 1
                if updated:
                    logger.info("Updated %d player team assignments from MLB API", updated)
            except Exception:
                logger.warning("Could not fetch live rosters from MLB API, using stored stints only")

        self._cache[season] = player_teams
        return player_teams
