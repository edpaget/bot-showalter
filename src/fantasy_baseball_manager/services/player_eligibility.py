from typing import TYPE_CHECKING

from fantasy_baseball_manager.models.zar.positions import build_position_map

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LeagueSettings, PitchingStats
    from fantasy_baseball_manager.repos import PitchingStatsRepo, PositionAppearanceRepo


class PlayerEligibilityService:
    """Provides batter and pitcher position data with season fallback.

    When position data is missing for the target season, falls back
    to the previous season (season - 1).
    """

    def __init__(
        self,
        position_repo: PositionAppearanceRepo,
        *,
        pitching_stats_repo: PitchingStatsRepo | None = None,
    ) -> None:
        self._position_repo = position_repo
        self._pitching_stats_repo = pitching_stats_repo

    def get_batter_positions(
        self,
        season: int,
        league: LeagueSettings,
    ) -> dict[int, list[str]]:
        """Return a map of player IDs to eligible league-settings position keys.

        Combines position data from *season* and prior seasons controlled by
        ``league.eligibility.carryover_seasons``. Each season's appearances are
        filtered by ``league.eligibility.batter_min_games`` independently.
        """
        appearances = list(self._position_repo.get_by_season(season))
        for i in range(1, league.eligibility.carryover_seasons + 1):
            appearances.extend(self._position_repo.get_by_season(season - i))
        return build_position_map(appearances, league, min_games=league.eligibility.batter_min_games)

    def get_pitcher_positions(
        self,
        season: int,
        league: LeagueSettings,
        pitcher_ids: list[int],
    ) -> dict[int, list[str]]:
        """Return a map of pitcher IDs to eligible pitcher position keys.

        If ``league.pitcher_positions`` is empty, returns ``["p"]`` for every
        pitcher (backward-compatible behaviour).

        Otherwise derives SP/RP eligibility from pitching stats (games started
        vs. relief appearances) and falls back to *season - 1* when no stats
        exist for the target season.
        """
        if not league.pitcher_positions:
            return {pid: ["p"] for pid in pitcher_ids}

        stats = self._get_pitching_stats(season, league.eligibility.carryover_seasons)
        aggregated = self._aggregate_pitching_stats(stats, pitcher_ids)
        return self._classify(
            aggregated,
            pitcher_ids,
            league,
            sp_min_starts=league.eligibility.sp_min_starts,
            rp_min_relief=league.eligibility.rp_min_relief,
        )

    def _get_pitching_stats(self, season: int, carryover_seasons: int) -> list[PitchingStats]:
        if self._pitching_stats_repo is None:
            return []
        result = list(self._pitching_stats_repo.get_by_season(season))
        for i in range(1, carryover_seasons + 1):
            result.extend(self._pitching_stats_repo.get_by_season(season - i))
        return result

    @staticmethod
    def _aggregate_pitching_stats(
        stats: list[PitchingStats],
        pitcher_ids: list[int],
    ) -> dict[int, tuple[int, int]]:
        """Aggregate max(g), max(gs) per player across sources."""
        pid_set = set(pitcher_ids)
        agg: dict[int, tuple[int, int]] = {}
        for s in stats:
            if s.player_id not in pid_set:
                continue
            g = s.g or 0
            gs = s.gs or 0
            if s.player_id in agg:
                prev_g, prev_gs = agg[s.player_id]
                agg[s.player_id] = (max(prev_g, g), max(prev_gs, gs))
            else:
                agg[s.player_id] = (g, gs)
        return agg

    @staticmethod
    def _classify(
        aggregated: dict[int, tuple[int, int]],
        pitcher_ids: list[int],
        league: LeagueSettings,
        *,
        sp_min_starts: int,
        rp_min_relief: int,
    ) -> dict[int, list[str]]:
        config = league.pitcher_positions
        result: dict[int, list[str]] = {}
        for pid in pitcher_ids:
            positions: list[str] = []
            if pid in aggregated:
                g, gs = aggregated[pid]
                if gs >= sp_min_starts and "sp" in config:
                    positions.append("sp")
                if (g - gs) >= rp_min_relief and "rp" in config:
                    positions.append("rp")
                if positions and "p" in config:
                    positions.append("p")
            if not positions and "p" in config:
                # Rookie / no stats — get flex "p" if available
                positions = ["p"]
            result[pid] = positions
        return result
