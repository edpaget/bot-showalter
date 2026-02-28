from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import Recommendation, RecommendationWeights
from fantasy_baseball_manager.services.draft_state import DraftEngine, DraftFormat, DraftState

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow


def recommend(
    state: DraftState,
    *,
    weights: RecommendationWeights | None = None,
    limit: int = 10,
) -> list[Recommendation]:
    """Return ranked draft recommendations from the current draft state.

    Pure function — no IO, no side effects.
    """
    w = weights or RecommendationWeights()
    pool = list(state.available_pool.values())
    if not pool:
        return []

    max_value = max(p.value for p in pool)
    if max_value <= 0:
        max_value = 1.0  # avoid division by zero

    needs = _user_needs(state)
    if not needs:
        return []

    scarcity = _compute_scarcity(state.available_pool, needs)
    picks_until = _picks_until_next(state)

    scored: list[tuple[float, DraftBoardRow]] = []
    for player in pool:
        if not _is_recommendable(player, needs):
            continue
        score = _score_player(player, w, max_value, needs, scarcity, pool, state, picks_until)
        scored.append((score, player))

    scored.sort(key=lambda t: t[0], reverse=True)

    results: list[Recommendation] = []
    for score, player in scored[:limit]:
        results.append(
            Recommendation(
                player_id=player.player_id,
                player_name=player.player_name,
                position=player.position,
                value=player.value,
                score=round(score, 4),
                reason=_build_reason(player, w, max_value, needs, scarcity, pool, state, picks_until),
            )
        )
    return results


def _user_needs(state: DraftState) -> dict[str, int]:
    """Compute unfilled roster slots for the user's team."""
    config = state.config
    needs: dict[str, int] = {}
    for pos, total in config.roster_slots.items():
        filled = sum(1 for p in state.team_rosters[config.user_team] if p.position == pos)
        remaining = total - filled
        if remaining > 0:
            needs[pos] = remaining
    return needs


def _is_recommendable(
    player: DraftBoardRow,
    needs: dict[str, int],
) -> bool:
    """Check if a player can fill any open roster slot."""
    # Direct position match
    if player.position in needs:
        return True
    # Flex: UTIL for batters, P for pitchers
    if player.player_type == "batter" and "UTIL" in needs:
        return True
    return player.player_type == "pitcher" and "P" in needs


def _need_bonus(player: DraftBoardRow, needs: dict[str, int]) -> float:
    """1.0 if the player's position has unfilled slots, 0.0 otherwise."""
    if player.position in needs:
        return 1.0
    return 0.0


_SCARCITY_DEPTH = 5


def _compute_scarcity(
    pool: dict[int, DraftBoardRow],
    needs: dict[str, int],
) -> dict[str, float]:
    """Per-position value dropoff from #1 to #5 available, normalized to [0, 1].

    Only computes scarcity for positions the user needs.
    Steep dropoff → high scarcity → draft now.
    Single player at a position → maximum scarcity (1.0).
    """
    scarcity: dict[str, float] = {}
    # Group available players by position
    by_pos: dict[str, list[float]] = {}
    for p in pool.values():
        if p.position in needs:
            by_pos.setdefault(p.position, []).append(p.value)

    # Sort each position's values descending
    for values in by_pos.values():
        values.sort(reverse=True)

    # Compute raw dropoffs for positions with multiple players
    raw_dropoffs: dict[str, float] = {}
    for pos, values in by_pos.items():
        top = values[:_SCARCITY_DEPTH]
        if len(top) > 1:
            raw_dropoffs[pos] = top[0] - top[-1]

    max_dropoff = max(raw_dropoffs.values()) if raw_dropoffs else 1.0
    if max_dropoff <= 0:
        max_dropoff = 1.0

    for pos, values in by_pos.items():
        top = values[:_SCARCITY_DEPTH]
        if len(top) <= 1:
            # Single player (or none) → maximum scarcity
            scarcity[pos] = 1.0
        else:
            scarcity[pos] = raw_dropoffs[pos] / max_dropoff

    # Positions in needs but not in pool → maximum scarcity
    for pos in needs:
        if pos not in scarcity and pos not in ("UTIL", "P"):
            scarcity[pos] = 1.0

    return scarcity


def _tier_urgency(player: DraftBoardRow, pool: list[DraftBoardRow]) -> float:
    """1.0 if next-best at position is in a different (worse) tier, 0.0 if same tier.

    Returns 0.5 if no tier data is available.
    """
    if player.tier is None:
        return 0.5

    # Find other players at the same position, sorted by value descending
    same_pos = sorted(
        [p for p in pool if p.position == player.position and p.player_id != player.player_id],
        key=lambda p: p.value,
        reverse=True,
    )

    # Find the next-best player (first one with value <= this player's value)
    next_best: DraftBoardRow | None = None
    for p in same_pos:
        if p.value <= player.value:
            next_best = p
            break

    if next_best is None:
        # This player is the worst at the position — no urgency
        return 0.0

    if next_best.tier is None:
        return 0.5

    if next_best.tier != player.tier:
        return 1.0
    return 0.0


def _picks_until_next(state: DraftState) -> int:
    """How many picks until the user selects again.

    Snake: compute from pick ordering. Auction: always 0.
    """
    if state.config.format == DraftFormat.AUCTION:
        return 0

    teams = state.config.teams
    user = state.config.user_team
    current = state.current_pick

    # Check if the user is on the clock right now
    if DraftEngine._snake_team(current, teams) == user:
        return 0

    # Scan forward to find the next pick for this user
    for offset in range(1, 2 * teams):
        future = current + offset
        if DraftEngine._snake_team(future, teams) == user:
            return offset

    return 2 * teams  # fallback — shouldn't happen


def _adp_availability(player: DraftBoardRow, picks_until: int, current_pick: int) -> float:
    """Score how likely a player is to be taken before user's next pick.

    Returns 1.0 if player's ADP suggests they'll be gone, 0.0 if safe.
    Returns 0.0 for auction or no ADP data.
    """
    if picks_until == 0 or player.adp_overall is None:
        return 0.0

    next_user_pick = current_pick + picks_until
    # If player's ADP is before our next pick, they're likely gone → urgency
    if player.adp_overall < next_user_pick:
        # Scale: the closer the ADP to current_pick, the more urgent
        urgency = (next_user_pick - player.adp_overall) / picks_until
        return min(1.0, max(0.0, urgency))
    return 0.0


def _score_player(
    player: DraftBoardRow,
    w: RecommendationWeights,
    max_value: float,
    needs: dict[str, int],
    scarcity: dict[str, float],
    pool: list[DraftBoardRow],
    state: DraftState,
    picks_until: int,
) -> float:
    """Compute composite recommendation score for a player."""
    value_norm = player.value / max_value
    need = _need_bonus(player, needs)
    scar = scarcity.get(player.position, 0.0)
    tier = _tier_urgency(player, pool)
    adp = _adp_availability(player, picks_until, state.current_pick)

    score = w.value * value_norm + w.need * need + w.scarcity * scar + w.tier * tier + w.adp * adp
    return score


def _build_reason(
    player: DraftBoardRow,
    w: RecommendationWeights,
    max_value: float,
    needs: dict[str, int],
    scarcity: dict[str, float],
    pool: list[DraftBoardRow],
    state: DraftState,
    picks_until: int,
) -> str:
    """Generate human-readable reason from the dominant secondary scoring factors."""
    pos = player.position

    # Compute weighted contributions for each secondary factor
    contributions: list[tuple[float, str]] = []

    need_val = _need_bonus(player, needs)
    if w.need > 0 and need_val > 0:
        remaining = needs.get(pos, 0)
        label = f"fills need at {pos} ({remaining} slot remaining)"
        contributions.append((w.need * need_val, label))

    scar_val = scarcity.get(pos, 0.0)
    if w.scarcity > 0 and scar_val > 0.3:
        contributions.append((w.scarcity * scar_val, f"positional scarcity at {pos}"))

    tier_val = _tier_urgency(player, pool)
    if w.tier > 0 and tier_val >= 0.8:
        tier_label = f"tier {player.tier}" if player.tier is not None else "tier break"
        contributions.append((w.tier * tier_val, f"tier urgency at {pos} ({tier_label})"))

    adp_val = _adp_availability(player, picks_until, state.current_pick)
    if w.adp > 0 and adp_val > 0:
        contributions.append((w.adp * adp_val, "ADP value: may be unavailable at next pick"))

    if not contributions:
        return "best value available"

    # Sort by weighted contribution descending, take top 2
    contributions.sort(key=lambda t: t[0], reverse=True)
    parts = [label for _, label in contributions[:2]]
    return " + ".join(parts)
