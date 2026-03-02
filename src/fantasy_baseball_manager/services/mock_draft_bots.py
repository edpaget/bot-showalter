"""Bot strategies for mock draft simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random

    from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
    from fantasy_baseball_manager.domain.league_settings import LeagueSettings
    from fantasy_baseball_manager.domain.mock_draft import DraftPick


class ADPBot:
    """Picks the player with the lowest ADP from available. None ADP treated as infinity."""

    def __init__(self, *, rng: random.Random) -> None:
        self._rng = rng

    def pick(
        self,
        available: list[DraftBoardRow],
        roster: list[DraftPick],
        league: LeagueSettings,
    ) -> int:
        best = min(available, key=lambda p: p.adp_overall if p.adp_overall is not None else float("inf"))
        return best.player_id


class BestValueBot:
    """Picks the highest-value player from available."""

    def __init__(self, *, rng: random.Random) -> None:
        self._rng = rng

    def pick(
        self,
        available: list[DraftBoardRow],
        roster: list[DraftPick],
        league: LeagueSettings,
    ) -> int:
        best = max(available, key=lambda p: p.value)
        return best.player_id


class PositionalNeedBot:
    """Prefers players whose primary position has an open slot. Falls back to highest value."""

    def __init__(self, *, rng: random.Random) -> None:
        self._rng = rng

    def pick(
        self,
        available: list[DraftBoardRow],
        roster: list[DraftPick],
        league: LeagueSettings,
    ) -> int:
        filled: dict[str, int] = {}
        for p in roster:
            filled[p.position] = filled.get(p.position, 0) + 1

        needed_positions: set[str] = set()
        for pos, count in league.positions.items():
            if filled.get(pos, 0) < count:
                needed_positions.add(pos)

        # Players at a needed position, sorted by value descending
        needed_players = [p for p in available if p.position in needed_positions]
        if needed_players:
            best = max(needed_players, key=lambda p: p.value)
            return best.player_id

        # Fallback: highest value overall
        best = max(available, key=lambda p: p.value)
        return best.player_id


class RandomBot:
    """Picks uniformly at random from the top 20 available players."""

    def __init__(self, *, rng: random.Random) -> None:
        self._rng = rng

    def pick(
        self,
        available: list[DraftBoardRow],
        roster: list[DraftPick],
        league: LeagueSettings,
    ) -> int:
        pool = available[:20] if len(available) > 20 else available
        chosen = self._rng.choice(pool)
        return chosen.player_id
