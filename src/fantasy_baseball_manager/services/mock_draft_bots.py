"""Bot strategies for mock draft simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import random

    from fantasy_baseball_manager.domain import DraftBoardRow, DraftPick, LeagueSettings


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


# ---------------------------------------------------------------------------
# Composable rule system
# ---------------------------------------------------------------------------


class StrategyRule(Protocol):
    """Protocol for scoring a player candidate during draft pick selection."""

    def score(
        self,
        player: DraftBoardRow,
        roster: list[DraftPick],
        round: int,
        league: LeagueSettings,
    ) -> float | None:
        """Return a score for the player, or None if this rule doesn't apply."""
        ...


@dataclass(frozen=True)
class WeightedRule:
    """A strategy rule paired with a weight for use in CompositeBot."""

    rule: StrategyRule
    weight: float


class TierValueRule:
    """Boosts score for players in tier 1-2. Returns None for tier 3+ or no tier."""

    def score(
        self,
        player: DraftBoardRow,
        roster: list[DraftPick],
        round: int,
        league: LeagueSettings,
    ) -> float | None:
        if player.tier is None or player.tier > 2:
            return None
        if player.tier == 1:
            return player.value * 2.0
        # tier 2
        return player.value * 1.0


class PositionTargetRule:
    """Boosts score for a specific position within a round window."""

    def __init__(self, *, position: str, rounds: tuple[int, int]) -> None:
        self._position = position
        self._round_start = rounds[0]
        self._round_end = rounds[1]

    def score(
        self,
        player: DraftBoardRow,
        roster: list[DraftPick],
        round: int,
        league: LeagueSettings,
    ) -> float | None:
        if player.position != self._position:
            return None
        if not (self._round_start <= round <= self._round_end):
            return None
        return player.value


class CategoryNeedRule:
    """Boosts score for players who improve the roster's weakest category z-score.

    Requires an external z-score lookup since DraftPick doesn't carry z-scores.
    """

    def __init__(self, *, z_score_lookup: dict[int, dict[str, float]]) -> None:
        self._z_score_lookup = z_score_lookup

    def score(
        self,
        player: DraftBoardRow,
        roster: list[DraftPick],
        round: int,
        league: LeagueSettings,
    ) -> float | None:
        if not roster:
            return None

        # Sum z-scores from roster players
        category_totals: dict[str, float] = {}
        for pick in roster:
            zscores = self._z_score_lookup.get(pick.player_id, {})
            for cat, z in zscores.items():
                category_totals[cat] = category_totals.get(cat, 0.0) + z

        if not category_totals:
            return None

        # Find the weakest category
        weakest_cat = min(category_totals, key=lambda c: category_totals[c])

        # Return the player's z-score in that category if positive
        player_z = player.category_z_scores.get(weakest_cat)
        if player_z is None or player_z <= 0:
            return None
        return player_z


class FallbackBestValueRule:
    """Always returns the player's raw value. Never returns None."""

    def score(
        self,
        player: DraftBoardRow,
        roster: list[DraftPick],
        round: int,
        league: LeagueSettings,
    ) -> float | None:
        return player.value


class CompositeBot:
    """Configurable bot that scores players using a weighted rule stack.

    Implements the DraftBot protocol. For each available player, computes the
    weighted sum of all applicable rule scores and picks the player with the
    highest total.
    """

    def __init__(self, *, rules: list[WeightedRule], rng: random.Random) -> None:
        self._rules = rules
        self._rng = rng

    def pick(
        self,
        available: list[DraftBoardRow],
        roster: list[DraftPick],
        league: LeagueSettings,
    ) -> int:
        round_num = len(roster) + 1

        best_id = available[0].player_id
        best_score = float("-inf")

        for player in available:
            total = 0.0
            any_scored = False
            for wr in self._rules:
                s = wr.rule.score(player, roster, round_num, league)
                if s is not None:
                    total += wr.weight * s
                    any_scored = True
            if any_scored and total > best_score:
                best_score = total
                best_id = player.player_id

        return best_id
