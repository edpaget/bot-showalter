from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import InjuryProfile
from fantasy_baseball_manager.name_utils import resolve_players
from fantasy_baseball_manager.services.games_lost_estimator import estimate_games_lost

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ExpectedGamesLost, ILStint
    from fantasy_baseball_manager.repos import ILStintRepo, PlayerRepo

# Default days for IL types when no days or dates are available
_IL_TYPE_DEFAULTS: dict[str, int] = {
    "10-day": 10,
    "15-day": 15,
    "60-day": 60,
    "7-day": 7,
}


def _compute_days(stint: ILStint) -> int:
    """Compute days lost for a stint: use days field, then date diff, then IL type default."""
    if stint.days is not None:
        return stint.days
    if stint.end_date is not None:
        try:
            start = date.fromisoformat(stint.start_date)
            end = date.fromisoformat(stint.end_date)
            diff = (end - start).days
            return max(diff, 0)
        except ValueError:
            pass
    return _IL_TYPE_DEFAULTS.get(stint.il_type, 15)


def build_profiles(
    il_stints: list[ILStint],
    seasons: list[int],
) -> dict[int, InjuryProfile]:
    """Build injury profiles from IL stint data.

    Pure function — takes pre-fetched stints and the season range.
    Returns dict[player_id, InjuryProfile].
    """
    season_set = set(seasons)
    n_seasons = len(seasons)
    if n_seasons == 0:
        return {}

    # Determine the last 2 seasons for recent_stints filtering
    sorted_seasons = sorted(seasons)
    recent_seasons = set(sorted_seasons[-2:])

    # Group stints by player
    by_player: dict[int, list[ILStint]] = defaultdict(list)
    for stint in il_stints:
        if stint.season in season_set:
            by_player[stint.player_id].append(stint)

    profiles: dict[int, InjuryProfile] = {}
    for player_id, stints in by_player.items():
        total_stints = len(stints)

        # Days lost per stint
        days_per_stint = [_compute_days(s) for s in stints]
        total_days_lost = sum(days_per_stint)

        # Days per season
        days_by_season: dict[int, int] = defaultdict(int)
        for stint, days in zip(stints, days_per_stint, strict=True):
            days_by_season[stint.season] += days

        avg_days_per_season = total_days_lost / n_seasons
        max_days_in_season = max(days_by_season.values()) if days_by_season else 0

        # Percent of seasons with at least one IL stint
        seasons_with_il = len(days_by_season)
        pct_seasons_with_il = seasons_with_il / n_seasons

        # Injury locations
        injury_locations: dict[str, int] = defaultdict(int)
        for stint in stints:
            if stint.injury_location:
                injury_locations[stint.injury_location] += 1

        # Recent stints (last 2 seasons)
        recent = [s for s in stints if s.season in recent_seasons]
        recent.sort(key=lambda s: s.start_date)

        profiles[player_id] = InjuryProfile(
            player_id=player_id,
            seasons_tracked=n_seasons,
            total_stints=total_stints,
            total_days_lost=total_days_lost,
            avg_days_per_season=avg_days_per_season,
            max_days_in_season=max_days_in_season,
            pct_seasons_with_il=pct_seasons_with_il,
            injury_locations=dict(injury_locations),
            recent_stints=recent,
        )

    return profiles


class InjuryProfiler:
    """Service for looking up injury profiles and listing high-risk players."""

    def __init__(self, player_repo: PlayerRepo, il_stint_repo: ILStintRepo) -> None:
        self._player_repo = player_repo
        self._il_stint_repo = il_stint_repo

    def lookup_profile(self, player_name: str, seasons: list[int]) -> tuple[InjuryProfile, str] | None:
        """Look up injury profile for a single player by name."""
        players = resolve_players(self._player_repo, player_name)
        if not players:
            return None

        player = players[0]
        assert player.id is not None  # noqa: S101 - type narrowing

        # Fetch stints for this player across all seasons
        stints: list[ILStint] = []
        for season in seasons:
            stints.extend(self._il_stint_repo.get_by_player_season(player.id, season))

        profiles = build_profiles(stints, seasons)
        profile = profiles.get(
            player.id,
            InjuryProfile(
                player_id=player.id,
                seasons_tracked=len(seasons),
                total_stints=0,
                total_days_lost=0,
                avg_days_per_season=0.0,
                max_days_in_season=0,
                pct_seasons_with_il=0.0,
            ),
        )

        name = f"{player.name_first} {player.name_last}"
        return (profile, name)

    def list_high_risk(
        self,
        seasons: list[int],
        *,
        min_stints: int = 1,
        top_n: int | None = None,
    ) -> list[tuple[InjuryProfile, str]]:
        """List most injury-prone players sorted by total_days_lost desc."""
        # Fetch all stints for each season
        all_stints: list[ILStint] = []
        for season in seasons:
            all_stints.extend(self._il_stint_repo.get_by_season(season))

        profiles = build_profiles(all_stints, seasons)

        # Filter by min_stints
        filtered = [(p, pid) for pid, p in profiles.items() if p.total_stints >= min_stints]

        # Sort by total_days_lost descending
        filtered.sort(key=lambda x: x[0].total_days_lost, reverse=True)

        if top_n is not None:
            filtered = filtered[:top_n]

        # Look up player names
        player_ids = [pid for _, pid in filtered]
        players = self._player_repo.get_by_ids(player_ids)
        name_map = {p.id: f"{p.name_first} {p.name_last}" for p in players}

        return [(profile, name_map.get(pid, f"Player {pid}")) for profile, pid in filtered]

    def estimate_player_games_lost(
        self,
        player_name: str,
        seasons: list[int],
        projection_season: int,
    ) -> tuple[ExpectedGamesLost, InjuryProfile, str] | None:
        """Estimate games lost for a single player by name."""
        result = self.lookup_profile(player_name, seasons)
        if result is None:
            return None
        profile, name = result
        estimate = estimate_games_lost(profile, projection_season)
        return (estimate, profile, name)

    def list_games_lost_estimates(
        self,
        seasons: list[int],
        projection_season: int,
        *,
        min_stints: int = 1,
        top_n: int | None = None,
    ) -> list[tuple[ExpectedGamesLost, InjuryProfile, str]]:
        """List players ranked by expected games lost."""
        high_risk = self.list_high_risk(seasons, min_stints=min_stints, top_n=None)
        results: list[tuple[ExpectedGamesLost, InjuryProfile, str]] = []
        for profile, name in high_risk:
            estimate = estimate_games_lost(profile, projection_season)
            results.append((estimate, profile, name))

        results.sort(key=lambda x: x[0].expected_days_lost, reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results
