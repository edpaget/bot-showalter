from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.domain import AvailabilityWindow, DraftPlan, Recommendation, RecommendationWeights
from fantasy_baseball_manager.services.draft_state import DraftEngine, DraftFormat, DraftState

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow


class CategoryBalanceFn(Protocol):
    """Callable that scores available players by how much they address weak categories."""

    def __call__(self, roster_ids: list[int], available_ids: list[int]) -> dict[int, float]: ...


def recommend(
    state: DraftState,
    *,
    weights: RecommendationWeights | None = None,
    limit: int = 10,
    category_balance_fn: CategoryBalanceFn | None = None,
    cat_scores: dict[int, float] | None = None,
    weak_categories: list[str] | None = None,
    draft_plan: DraftPlan | None = None,
    availability: list[AvailabilityWindow] | None = None,
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
    current_round = (state.current_pick - 1) // state.config.teams + 1

    # Build availability lookup once
    avail_map: dict[int, AvailabilityWindow] = {}
    if availability is not None:
        avail_map = {aw.player_id: aw for aw in availability}

    # Use pre-computed category balance scores, or compute them from the function
    if cat_scores is None and category_balance_fn is not None:
        roster_ids = [p.player_id for p in state.team_rosters[state.config.user_team]]
        available_ids = list(state.available_pool.keys())
        cat_scores = category_balance_fn(roster_ids, available_ids)
    effective_cat_scores: dict[int, float] = cat_scores or {}

    scored: list[tuple[float, DraftBoardRow]] = []
    for player in pool:
        if not _is_recommendable(player, needs):
            continue
        score = _score_player(
            player,
            w,
            max_value,
            needs,
            scarcity,
            pool,
            state,
            picks_until,
            effective_cat_scores,
            draft_plan=draft_plan,
            current_round=current_round,
            avail_map=avail_map,
        )
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
                reason=_build_reason(
                    player,
                    w,
                    max_value,
                    needs,
                    scarcity,
                    pool,
                    state,
                    picks_until,
                    effective_cat_scores,
                    draft_plan=draft_plan,
                    current_round=current_round,
                    avail_map=avail_map,
                    weak_categories=weak_categories,
                ),
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


def _mock_position_bonus(
    player: DraftBoardRow,
    draft_plan: DraftPlan | None,
    current_round: int,
) -> float:
    """Return confidence bonus if player's position matches the plan target for this round."""
    if draft_plan is None:
        return 0.0
    for target in draft_plan.targets:
        start, end = target.round_range
        if start <= current_round <= end and player.position == target.position:
            return target.confidence
    return 0.0


def _mock_availability_score(
    player: DraftBoardRow,
    avail_map: dict[int, AvailabilityWindow],
    picks_until: int,
) -> float:
    """Return urgency/wait signal from mock simulation availability data."""
    if not avail_map or picks_until == 0:
        return 0.0
    window = avail_map.get(player.player_id)
    if window is None:
        return 0.0
    prob = window.available_at_user_pick
    if prob >= 0.8:
        return -0.5
    if prob < 0.3:
        return 1.0
    # Linear interpolation: prob in [0.3, 0.8] → score in [1.0, -0.5]
    return 1.0 + (prob - 0.3) * (-0.5 - 1.0) / (0.8 - 0.3)


def _score_player(
    player: DraftBoardRow,
    w: RecommendationWeights,
    max_value: float,
    needs: dict[str, int],
    scarcity: dict[str, float],
    pool: list[DraftBoardRow],
    state: DraftState,
    picks_until: int,
    cat_scores: dict[int, float] | None = None,
    *,
    draft_plan: DraftPlan | None = None,
    current_round: int = 1,
    avail_map: dict[int, AvailabilityWindow] | None = None,
) -> float:
    """Compute composite recommendation score for a player."""
    value_norm = player.value / max_value
    need = _need_bonus(player, needs)
    scar = scarcity.get(player.position, 0.0)
    tier = _tier_urgency(player, pool)
    adp = _adp_availability(player, picks_until, state.current_pick)
    cat_bal = cat_scores.get(player.player_id, 0.0) if cat_scores else 0.0
    mock_pos = _mock_position_bonus(player, draft_plan, current_round)
    mock_avail = _mock_availability_score(player, avail_map or {}, picks_until)

    score = (
        w.value * value_norm
        + w.need * need
        + w.scarcity * scar
        + w.tier * tier
        + w.adp * adp
        + w.category_balance * cat_bal
        + w.mock_position * mock_pos
        + w.mock_availability * mock_avail
    )
    return score


def _build_reason(
    player: DraftBoardRow,
    w: RecommendationWeights,
    max_value: float,  # noqa: ARG001
    needs: dict[str, int],
    scarcity: dict[str, float],
    pool: list[DraftBoardRow],
    state: DraftState,
    picks_until: int,
    cat_scores: dict[int, float] | None = None,
    *,
    draft_plan: DraftPlan | None = None,
    current_round: int = 1,
    avail_map: dict[int, AvailabilityWindow] | None = None,
    weak_categories: list[str] | None = None,
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

    cat_bal = cat_scores.get(player.player_id, 0.0) if cat_scores else 0.0
    if w.category_balance > 0 and cat_bal > 0.3:
        cat_label = "fills " + " + ".join(weak_categories) + " gaps" if weak_categories else "addresses weak categories"
        contributions.append((w.category_balance * cat_bal, cat_label))

    mock_pos = _mock_position_bonus(player, draft_plan, current_round)
    if w.mock_position > 0 and mock_pos > 0:
        contributions.append((w.mock_position * mock_pos, f"mock plan targets {pos} this round"))

    mock_avail = _mock_availability_score(player, avail_map or {}, picks_until)
    if w.mock_availability > 0 and mock_avail > 0.5:
        contributions.append((w.mock_availability * mock_avail, "mock sims: likely gone before next pick"))

    if not contributions:
        return "best value available"

    # Sort by weighted contribution descending, take top 2
    contributions.sort(key=lambda t: t[0], reverse=True)
    parts = [label for _, label in contributions[:2]]
    return " + ".join(parts)
