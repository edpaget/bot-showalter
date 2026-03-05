from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerUniverseProvider

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import BattingStatsRepo, PitchingStatsRepo

# Re-export for backward compatibility
__all__ = ["PlayerUniverseProvider", "StatsBasedPlayerUniverse"]


class StatsBasedPlayerUniverse:
    def __init__(self, batting_repo: BattingStatsRepo, pitching_repo: PitchingStatsRepo) -> None:
        self._batting_repo = batting_repo
        self._pitching_repo = pitching_repo

    def get_player_ids(
        self,
        season: int,
        player_type: str,
        *,
        source: str | None = None,
        min_pa: int | None = None,
        min_ip: float | None = None,
    ) -> set[int]:
        prior = season - 1
        if player_type == "batter":
            stats = self._batting_repo.get_by_season(prior, source)
            if min_pa is not None:
                stats = [s for s in stats if s.pa is not None and s.pa >= min_pa]
            return {s.player_id for s in stats}
        else:  # pitcher
            stats = self._pitching_repo.get_by_season(prior, source)
            if min_ip is not None:
                stats = [s for s in stats if s.ip is not None and s.ip >= min_ip]
            return {s.player_id for s in stats}
