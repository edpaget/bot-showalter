from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.models.zar.positions import build_position_map
from fantasy_baseball_manager.repos.protocols import PositionAppearanceRepo


class PlayerEligibilityService:
    """Provides batter position data with season fallback.

    When position data is missing for the target season, falls back
    to the previous season (season - 1).
    """

    def __init__(self, position_repo: PositionAppearanceRepo) -> None:
        self._position_repo = position_repo

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
