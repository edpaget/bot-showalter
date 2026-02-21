from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.models.zar.positions import build_position_map
from fantasy_baseball_manager.repos.protocols import PitchingStatsRepo, PositionAppearanceRepo


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
        *,
        min_games: int = 10,
    ) -> dict[int, list[str]]:
        """Return a map of player IDs to eligible league-settings position keys.

        If no position data exists for *season*, falls back to *season - 1*.
        Positions with fewer than *min_games* appearances are excluded.
        """
        appearances = self._position_repo.get_by_season(season)
        if not appearances:
            appearances = self._position_repo.get_by_season(season - 1)
        return build_position_map(appearances, league, min_games=min_games)

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

        stats = self._get_pitching_stats(season)
        aggregated = self._aggregate_pitching_stats(stats, pitcher_ids)
        return self._classify(aggregated, pitcher_ids, league)

    def _get_pitching_stats(self, season: int) -> list[PitchingStats]:
        if self._pitching_stats_repo is None:
            return []
        stats = self._pitching_stats_repo.get_by_season(season)
        if not stats:
            stats = self._pitching_stats_repo.get_by_season(season - 1)
        return stats

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
    ) -> dict[int, list[str]]:
        config = league.pitcher_positions
        result: dict[int, list[str]] = {}
        for pid in pitcher_ids:
            positions: list[str] = []
            if pid in aggregated:
                g, gs = aggregated[pid]
                if gs > 0 and "sp" in config:
                    positions.append("sp")
                if (g - gs) > 0 and "rp" in config:
                    positions.append("rp")
                if positions and "p" in config:
                    positions.append("p")
            if not positions:
                # Rookie / no stats — get flex "p" if available
                if "p" in config:
                    positions = ["p"]
            result[pid] = positions
        return result
