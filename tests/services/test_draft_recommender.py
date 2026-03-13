from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.draft_plan import AvailabilityWindow, DraftPlan, DraftPlanTarget
from fantasy_baseball_manager.domain.draft_recommendation import (
    Recommendation,
    RecommendationWeights,
)
from fantasy_baseball_manager.services.draft_recommender import (
    _compute_scarcity,
    _compute_tier_urgency_map,
    _picks_until_next,
    recommend,
)
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftPick,
    DraftState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_player(
    player_id: int,
    name: str,
    position: str,
    value: float,
    *,
    player_type: str = "batter",
    tier: int | None = None,
    adp_overall: float | None = None,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type=player_type,
        position=position,
        value=value,
        category_z_scores={},
        tier=tier,
        adp_overall=adp_overall,
    )


def _make_state(
    players: list[DraftBoardRow],
    roster_slots: dict[str, int] | None = None,
    *,
    fmt: DraftFormat = DraftFormat.SNAKE,
    teams: int = 4,
    user_team: int = 1,
    picks: list[DraftPick] | None = None,
    budget: int = 0,
) -> DraftState:
    slots = roster_slots or {"C": 1, "1B": 1, "OF": 2, "P": 1}
    config = DraftConfig(
        teams=teams,
        roster_slots=slots,
        format=fmt,
        user_team=user_team,
        season=2026,
        budget=budget,
    )
    pool = {(p.player_id, p.player_type): p for p in players}
    pick_list = picks or []
    # Build team rosters from picks
    rosters: dict[int, list[DraftPick]] = {t: [] for t in range(1, teams + 1)}
    for pick in pick_list:
        rosters[pick.team].append(pick)
        # Remove picked player from pool by finding matching key
        for key in list(pool):
            if key[0] == pick.player_id:
                del pool[key]
                break
    budgets: dict[int, int] = {t: budget if fmt == DraftFormat.AUCTION else 0 for t in range(1, teams + 1)}
    return DraftState(
        config=config,
        picks=pick_list,
        available_pool=pool,
        team_rosters=rosters,
        team_budgets=budgets,
        current_pick=len(pick_list) + 1,
    )


# ---------------------------------------------------------------------------
# Step 1: Domain models — frozen behavior and defaults
# ---------------------------------------------------------------------------


class TestDomainModels:
    def test_recommendation_weights_defaults(self) -> None:
        w = RecommendationWeights()
        assert w.value == 1.0
        assert w.need == 0.3
        assert w.scarcity == 0.4
        assert w.tier == 0.2
        assert w.adp == 0.15
        assert w.category_balance == 0.25
        assert w.mock_position == 0.3
        assert w.mock_availability == 0.2

    def test_recommendation_weights_custom(self) -> None:
        w = RecommendationWeights(value=2.0, need=0.5, scarcity=0.0, tier=0.1, adp=0.0)
        assert w.value == 2.0
        assert w.need == 0.5
        assert w.scarcity == 0.0

    def test_recommendation_weights_frozen(self) -> None:
        w = RecommendationWeights()
        with pytest.raises(FrozenInstanceError):
            w.value = 5.0  # type: ignore[misc]

    def test_recommendation_fields(self) -> None:
        r = Recommendation(
            player_id=1,
            player_name="Mike Trout",
            position="OF",
            value=45.0,
            score=1.8,
            reason="best value available",
        )
        assert r.player_id == 1
        assert r.player_name == "Mike Trout"
        assert r.position == "OF"
        assert r.value == 45.0
        assert r.score == 1.8
        assert r.reason == "best value available"

    def test_recommendation_frozen(self) -> None:
        r = Recommendation(
            player_id=1,
            player_name="Mike Trout",
            position="OF",
            value=45.0,
            score=1.8,
            reason="best value available",
        )
        with pytest.raises(FrozenInstanceError):
            r.score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Step 2: Basic value scoring
# ---------------------------------------------------------------------------

POOL = [
    _make_player(1, "Player A", "C", 30.0),
    _make_player(2, "Player B", "1B", 25.0),
    _make_player(3, "Player C", "OF", 20.0),
    _make_player(4, "Player D", "OF", 18.0),
    _make_player(5, "Player E", "SP", 15.0, player_type="pitcher"),
    _make_player(6, "Player F", "C", 12.0),
    _make_player(7, "Player G", "OF", 10.0),
    _make_player(8, "Player H", "1B", 8.0),
]


class TestValueScoring:
    def test_highest_value_ranks_first(self) -> None:
        state = _make_state(POOL)
        recs = recommend(state)
        assert recs[0].player_id == 1
        assert recs[0].value == 30.0

    def test_limit_returns_n_results(self) -> None:
        state = _make_state(POOL)
        recs = recommend(state, limit=3)
        assert len(recs) == 3

    def test_empty_pool_returns_empty(self) -> None:
        state = _make_state([])
        recs = recommend(state)
        assert recs == []

    def test_returns_recommendation_type(self) -> None:
        state = _make_state(POOL)
        recs = recommend(state, limit=1)
        assert isinstance(recs[0], Recommendation)

    def test_value_only_ordering(self) -> None:
        """With value-only weights, ordering matches value descending."""
        w = RecommendationWeights(value=1.0, need=0.0, scarcity=0.0, tier=0.0, adp=0.0)
        state = _make_state(POOL)
        recs = recommend(state, weights=w)
        values = [r.value for r in recs]
        assert values == sorted(values, reverse=True)

    def test_default_limit_is_10(self) -> None:
        """Default limit caps results at 10."""
        many_players = [_make_player(i, f"Player {i}", "OF", float(100 - i)) for i in range(1, 30)]
        state = _make_state(many_players, roster_slots={"OF": 20})
        recs = recommend(state)
        assert len(recs) == 10


# ---------------------------------------------------------------------------
# Step 3: Need bonus
# ---------------------------------------------------------------------------


class TestNeedBonus:
    def test_unfilled_position_boosted_over_filled(self) -> None:
        """A lower-value player at an unfilled position outranks higher-value at filled.

        1B slot is full, but UTIL is open so the 1B player is still recommendable
        via flex — just without the need bonus. C slot is open so the catcher
        gets the need bonus and outranks despite lower raw value.
        """
        players = [
            _make_player(1, "Catcher", "C", 20.0),
            _make_player(2, "First Base", "1B", 22.0),
        ]
        pick = DraftPick(pick_number=1, team=1, player_id=99, player_name="X", position="1B")
        state = _make_state(
            players,
            roster_slots={"C": 1, "1B": 1, "UTIL": 1},
            picks=[pick],
        )
        w = RecommendationWeights(value=1.0, need=1.0, scarcity=0.0, tier=0.0, adp=0.0)
        recs = recommend(state, weights=w)
        # C should be boosted above 1B despite lower raw value
        assert recs[0].player_id == 1

    def test_filled_position_excluded(self) -> None:
        """Player at a completely filled position is excluded."""
        players = [
            _make_player(1, "Catcher A", "C", 30.0),
            _make_player(2, "OF Player", "OF", 20.0),
        ]
        pick = DraftPick(pick_number=1, team=1, player_id=99, player_name="X", position="C")
        state = _make_state(
            players,
            roster_slots={"C": 1, "OF": 1},
            picks=[pick],
        )
        recs = recommend(state)
        ids = [r.player_id for r in recs]
        assert 1 not in ids
        assert 2 in ids

    def test_multi_slot_position_partially_filled(self) -> None:
        """Position with 2 slots: still recommendable when 1 is filled."""
        players = [
            _make_player(1, "OF-1", "OF", 25.0),
            _make_player(2, "OF-2", "OF", 20.0),
        ]
        pick = DraftPick(pick_number=1, team=1, player_id=99, player_name="Picked OF", position="OF")
        state = _make_state(
            players,
            roster_slots={"OF": 2},
            picks=[pick],
        )
        recs = recommend(state)
        assert len(recs) == 2

    def test_flex_util_keeps_batter_recommendable(self) -> None:
        """Batter at filled position still recommended if UTIL slot is open."""
        players = [
            _make_player(1, "Catcher A", "C", 30.0),
        ]
        pick = DraftPick(pick_number=1, team=1, player_id=99, player_name="X", position="C")
        state = _make_state(
            players,
            roster_slots={"C": 1, "UTIL": 1},
            picks=[pick],
        )
        recs = recommend(state)
        assert len(recs) == 1
        assert recs[0].player_id == 1

    def test_flex_p_keeps_pitcher_recommendable(self) -> None:
        """Pitcher at filled SP slot still recommended if P flex slot is open."""
        players = [
            _make_player(1, "Pitcher A", "SP", 25.0, player_type="pitcher"),
        ]
        pick = DraftPick(pick_number=1, team=1, player_id=99, player_name="X", position="SP")
        state = _make_state(
            players,
            roster_slots={"SP": 1, "P": 1},
            picks=[pick],
        )
        recs = recommend(state)
        assert len(recs) == 1

    def test_no_needs_returns_empty(self) -> None:
        """All slots filled → no recommendations."""
        players = [_make_player(1, "Player A", "C", 30.0)]
        pick = DraftPick(pick_number=1, team=1, player_id=99, player_name="X", position="C")
        state = _make_state(
            players,
            roster_slots={"C": 1},
            picks=[pick],
        )
        recs = recommend(state)
        assert recs == []


# ---------------------------------------------------------------------------
# Step 4: Scarcity scoring
# ---------------------------------------------------------------------------


class TestScarcity:
    def test_single_player_at_position_max_scarcity(self) -> None:
        """Only one C available → scarcity should be 1.0 (maximum)."""

        players = [
            _make_player(1, "Only Catcher", "C", 30.0),
            _make_player(2, "OF-1", "OF", 25.0),
            _make_player(3, "OF-2", "OF", 24.0),
            _make_player(4, "OF-3", "OF", 23.0),
            _make_player(5, "OF-4", "OF", 22.0),
            _make_player(6, "OF-5", "OF", 21.0),
        ]
        pool = {(p.player_id, p.player_type): p for p in players}
        scarcity = _compute_scarcity(pool, {"C": 1, "OF": 2})
        assert scarcity["C"] == 1.0

    def test_flat_pool_low_scarcity(self) -> None:
        """OFs have similar values → low scarcity relative to SS with steep dropoff."""

        players = [
            _make_player(1, "OF-1", "OF", 20.0),
            _make_player(2, "OF-2", "OF", 19.5),
            _make_player(3, "OF-3", "OF", 19.0),
            _make_player(4, "OF-4", "OF", 18.5),
            _make_player(5, "OF-5", "OF", 18.0),
            # SS with steep dropoff for comparison
            _make_player(10, "SS-1", "SS", 35.0),
            _make_player(11, "SS-2", "SS", 15.0),
            _make_player(12, "SS-3", "SS", 10.0),
            _make_player(13, "SS-4", "SS", 5.0),
            _make_player(14, "SS-5", "SS", 2.0),
        ]
        pool = {(p.player_id, p.player_type): p for p in players}
        scarcity = _compute_scarcity(pool, {"OF": 2, "SS": 1})
        assert scarcity["OF"] < 0.3
        assert scarcity["SS"] > 0.7

    def test_steep_dropoff_high_scarcity(self) -> None:
        """Big gap between #1 and #5 at a position → high scarcity."""

        players = [
            _make_player(1, "SS-1", "SS", 40.0),
            _make_player(2, "SS-2", "SS", 15.0),
            _make_player(3, "SS-3", "SS", 10.0),
            _make_player(4, "SS-4", "SS", 5.0),
            _make_player(5, "SS-5", "SS", 2.0),
        ]
        pool = {(p.player_id, p.player_type): p for p in players}
        scarcity = _compute_scarcity(pool, {"SS": 1})
        assert scarcity["SS"] > 0.7

    def test_scarcity_boosts_scarce_position(self) -> None:
        """Scarce-position player outranks similar-value player at deep position.

        C has 1 player → max scarcity. OF has 5 with flat dropoff → low scarcity.
        SS has steep dropoff → high scarcity (provides normalization contrast).
        """
        players = [
            _make_player(1, "Only Catcher", "C", 20.0),
            _make_player(2, "OF-1", "OF", 21.0),
            _make_player(3, "OF-2", "OF", 20.5),
            _make_player(4, "OF-3", "OF", 20.0),
            _make_player(5, "OF-4", "OF", 19.5),
            _make_player(6, "OF-5", "OF", 19.0),
            # SS with steep dropoff — gives normalization contrast
            _make_player(10, "SS-1", "SS", 20.0),
            _make_player(11, "SS-2", "SS", 8.0),
            _make_player(12, "SS-3", "SS", 5.0),
            _make_player(13, "SS-4", "SS", 3.0),
            _make_player(14, "SS-5", "SS", 1.0),
        ]
        state = _make_state(players, roster_slots={"C": 1, "OF": 3, "SS": 1})
        w = RecommendationWeights(value=1.0, need=0.0, scarcity=1.0, tier=0.0, adp=0.0)
        recs = recommend(state, weights=w)
        # Catcher should rank higher than OF-1 despite slightly lower value
        assert recs[0].player_id == 1

    def test_scarcity_only_for_needed_positions(self) -> None:
        """Scarcity is only computed for positions the user needs."""

        players = [
            _make_player(1, "C-1", "C", 30.0),
            _make_player(2, "1B-1", "1B", 25.0),
        ]
        pool = {(p.player_id, p.player_type): p for p in players}
        scarcity = _compute_scarcity(pool, {"C": 1})
        assert "C" in scarcity
        assert "1B" not in scarcity


# ---------------------------------------------------------------------------
# Step 5: Tier urgency
# ---------------------------------------------------------------------------


class TestComputeTierUrgencyMap:
    def test_basic_tier_urgency_map(self) -> None:
        """Pre-computed map matches per-player _tier_urgency semantics."""
        players = [
            _make_player(1, "SS-1", "SS", 30.0, tier=1),
            _make_player(2, "SS-2", "SS", 25.0, tier=2),
            _make_player(3, "SS-3", "SS", 20.0, tier=2),
            _make_player(4, "C-1", "C", 28.0, tier=1),
            _make_player(5, "C-2", "C", 22.0, tier=1),
        ]
        result = _compute_tier_urgency_map(players)
        # SS-1: next-best is SS-2 (tier 2, different) → 1.0
        assert result[1] == 1.0
        # SS-2: next-best is SS-3 (tier 2, same) → 0.0
        assert result[2] == 0.0
        # SS-3: worst at position, no next-best → 0.0
        assert result[3] == 0.0
        # C-1: next-best is C-2 (tier 1, same) → 0.0
        assert result[4] == 0.0
        # C-2: worst at position → 0.0
        assert result[5] == 0.0

    def test_no_tier_data_returns_neutral(self) -> None:
        """Players without tier data get 0.5."""
        players = [
            _make_player(1, "SS-1", "SS", 30.0),
            _make_player(2, "SS-2", "SS", 25.0),
        ]
        result = _compute_tier_urgency_map(players)
        assert result[1] == 0.5
        assert result[2] == 0.5

    def test_single_player_at_position(self) -> None:
        """Sole player at a position → 0.0 (no next-best)."""
        players = [_make_player(1, "C-1", "C", 30.0, tier=1)]
        result = _compute_tier_urgency_map(players)
        assert result[1] == 0.0


class TestTierUrgency:
    def test_different_tier_high_urgency(self) -> None:
        """Next-best at position is in a worse tier → urgency = 1.0."""
        players = [
            _make_player(1, "SS-1", "SS", 30.0, tier=1),
            _make_player(2, "SS-2", "SS", 25.0, tier=2),
            _make_player(3, "SS-3", "SS", 20.0, tier=2),
        ]
        state = _make_state(players, roster_slots={"SS": 1})
        w = RecommendationWeights(value=0.0, need=0.0, scarcity=0.0, tier=1.0, adp=0.0)
        recs = recommend(state, weights=w)
        # SS-1 has tier urgency (last in tier 1), should score highest
        assert recs[0].player_id == 1
        assert recs[0].score == 1.0

    def test_same_tier_no_urgency(self) -> None:
        """Player whose next-best is in the same tier → urgency = 0.0."""
        players = [
            _make_player(1, "SS-1", "SS", 30.0, tier=1),
            _make_player(2, "SS-2", "SS", 28.0, tier=1),
            _make_player(3, "SS-3", "SS", 20.0, tier=2),
        ]
        state = _make_state(players, roster_slots={"SS": 1})
        w = RecommendationWeights(value=0.0, need=0.0, scarcity=0.0, tier=1.0, adp=0.0)
        recs = recommend(state, weights=w)
        ss1 = next(r for r in recs if r.player_id == 1)
        # SS-1's next-best is SS-2 (same tier 1) → urgency 0.0
        assert ss1.score == 0.0
        # SS-2's next-best is SS-3 (tier 2) → urgency 1.0
        ss2 = next(r for r in recs if r.player_id == 2)
        assert ss2.score == 1.0

    def test_no_tier_data_neutral(self) -> None:
        """No tier data → urgency = 0.5 (neutral)."""
        players = [
            _make_player(1, "SS-1", "SS", 30.0),
            _make_player(2, "SS-2", "SS", 25.0),
        ]
        state = _make_state(players, roster_slots={"SS": 1})
        w = RecommendationWeights(value=0.0, need=0.0, scarcity=0.0, tier=1.0, adp=0.0)
        recs = recommend(state, weights=w)
        assert recs[0].score == 0.5

    def test_tier_urgency_boosts_last_in_tier(self) -> None:
        """Tier urgency + value: lower-value player in a tier break beats same-tier peer."""
        players = [
            # SS: two in tier 1, one in tier 2
            _make_player(1, "SS-1", "SS", 30.0, tier=1),
            _make_player(2, "SS-2", "SS", 28.0, tier=1),
            _make_player(3, "SS-3", "SS", 20.0, tier=2),
            # C: all in tier 1 — no urgency
            _make_player(4, "C-1", "C", 29.0, tier=1),
            _make_player(5, "C-2", "C", 27.0, tier=1),
        ]
        state = _make_state(players, roster_slots={"SS": 1, "C": 1})
        w = RecommendationWeights(value=1.0, need=0.0, scarcity=0.0, tier=0.5, adp=0.0)
        recs = recommend(state, weights=w)
        # SS-1 (value=30, tier urgency=0 since SS-2 is same tier) scores
        # differently from SS-2 (value=28, tier urgency=1 since SS-3 is tier 2)
        ss2 = next(r for r in recs if r.player_id == 2)
        c1 = next(r for r in recs if r.player_id == 4)
        # SS-2 has tier urgency, C-1 doesn't, and they're close in value
        assert ss2.score > c1.score


# ---------------------------------------------------------------------------
# Step 6: ADP availability
# ---------------------------------------------------------------------------


class TestADPAvailability:
    def test_picks_until_next_snake(self) -> None:
        """In a 4-team snake, team 1 picks at 1, 8, 9, 16, ..."""

        state = _make_state(POOL, teams=4, user_team=1)
        # Pick 1 is user's. Next pick for user is pick 8.
        state.current_pick = 2  # just made pick 1
        assert _picks_until_next(state) == 6  # picks 2,3,4,5,6,7 → 6 picks away

    def test_picks_until_next_snake_round2(self) -> None:
        """After round 2 pick (pick 8 for team 1), next is pick 9."""

        state = _make_state(POOL, teams=4, user_team=1)
        state.current_pick = 9  # just made pick 8, now at pick 9
        # Team 1 also picks at 9 in snake (round 3 starts ascending again)
        assert _picks_until_next(state) == 0  # user is on the clock

    def test_picks_until_next_auction_returns_zero(self) -> None:
        """Auction format: always 0 (anyone can pick anytime)."""

        state = _make_state(POOL, fmt=DraftFormat.AUCTION, teams=2, budget=100)
        assert _picks_until_next(state) == 0

    def test_adp_boost_when_player_likely_gone(self) -> None:
        """Player whose ADP < current_pick + picks_until_next gets ADP boost."""
        players = [
            _make_player(1, "Bargain", "C", 20.0, adp_overall=5.0),
            _make_player(2, "Safe", "1B", 22.0, adp_overall=50.0),
        ]
        state = _make_state(players, teams=4, user_team=1)
        state.current_pick = 2  # just made pick 1, next user pick is 8
        w = RecommendationWeights(value=1.0, need=0.0, scarcity=0.0, tier=0.0, adp=1.0)
        recs = recommend(state, weights=w)
        # Bargain has ADP 5, likely gone before pick 8 → high adp score
        bargain = next(r for r in recs if r.player_id == 1)
        safe = next(r for r in recs if r.player_id == 2)
        assert bargain.score > safe.score

    def test_no_adp_data_neutral(self) -> None:
        """Player with no ADP data gets 0 ADP score."""
        players = [
            _make_player(1, "No ADP", "C", 20.0),
        ]
        state = _make_state(players, roster_slots={"C": 1})
        w = RecommendationWeights(value=0.0, need=0.0, scarcity=0.0, tier=0.0, adp=1.0)
        recs = recommend(state, weights=w)
        assert recs[0].score == 0.0

    def test_auction_no_adp_boost(self) -> None:
        """In auction, ADP availability score is always 0."""
        players = [
            _make_player(1, "Bargain", "C", 20.0, adp_overall=1.0),
        ]
        state = _make_state(
            players,
            roster_slots={"C": 1},
            fmt=DraftFormat.AUCTION,
            teams=2,
            budget=100,
        )
        w = RecommendationWeights(value=0.0, need=0.0, scarcity=0.0, tier=0.0, adp=1.0)
        recs = recommend(state, weights=w)
        assert recs[0].score == 0.0


# ---------------------------------------------------------------------------
# Step 7: Reason strings
# ---------------------------------------------------------------------------


class TestReasonStrings:
    def test_best_value_when_value_dominates(self) -> None:
        """When value is the only factor, reason is 'best value available'."""
        players = [_make_player(1, "Star", "C", 30.0)]
        state = _make_state(players, roster_slots={"C": 1})
        w = RecommendationWeights(value=1.0, need=0.0, scarcity=0.0, tier=0.0, adp=0.0)
        recs = recommend(state, weights=w)
        assert recs[0].reason == "best value available"

    def test_fills_need_reason(self) -> None:
        """Need is dominant secondary factor → reason mentions filling need."""
        players = [
            _make_player(1, "Catcher", "C", 15.0),
            _make_player(2, "OF", "OF", 20.0),
        ]
        # OF slot filled, C still open — C gets need bonus
        pick = DraftPick(pick_number=1, team=1, player_id=99, player_name="X", position="OF")
        state = _make_state(
            players,
            roster_slots={"C": 1, "OF": 1, "UTIL": 1},
            picks=[pick],
        )
        w = RecommendationWeights(value=0.5, need=1.0, scarcity=0.0, tier=0.0, adp=0.0)
        recs = recommend(state, weights=w)
        catcher = next(r for r in recs if r.player_id == 1)
        assert "fills need at C" in catcher.reason

    def test_scarcity_reason(self) -> None:
        """Scarcity is dominant secondary factor → reason mentions scarcity."""
        players = [
            _make_player(1, "Only C", "C", 20.0),
            # Deep SS pool for normalization contrast
            _make_player(10, "SS-1", "SS", 30.0),
            _make_player(11, "SS-2", "SS", 12.0),
            _make_player(12, "SS-3", "SS", 8.0),
            _make_player(13, "SS-4", "SS", 4.0),
            _make_player(14, "SS-5", "SS", 2.0),
        ]
        state = _make_state(players, roster_slots={"C": 1, "SS": 1})
        w = RecommendationWeights(value=0.5, need=0.0, scarcity=1.0, tier=0.0, adp=0.0)
        recs = recommend(state, weights=w)
        catcher = next(r for r in recs if r.player_id == 1)
        assert "positional scarcity at C" in catcher.reason

    def test_tier_urgency_reason(self) -> None:
        """Tier urgency is dominant secondary factor → reason mentions tier."""
        players = [
            _make_player(1, "SS-1", "SS", 20.0, tier=1),
            _make_player(2, "SS-2", "SS", 15.0, tier=2),
        ]
        state = _make_state(players, roster_slots={"SS": 1})
        w = RecommendationWeights(value=0.5, need=0.0, scarcity=0.0, tier=1.0, adp=0.0)
        recs = recommend(state, weights=w)
        ss1 = next(r for r in recs if r.player_id == 1)
        assert "tier urgency at SS" in ss1.reason

    def test_adp_reason(self) -> None:
        """ADP is dominant secondary factor → reason mentions ADP."""
        players = [
            _make_player(1, "Bargain", "C", 20.0, adp_overall=3.0),
        ]
        state = _make_state(players, roster_slots={"C": 1}, teams=4, user_team=1)
        state.current_pick = 2  # next user pick is 8
        w = RecommendationWeights(value=0.5, need=0.0, scarcity=0.0, tier=0.0, adp=1.0)
        recs = recommend(state, weights=w)
        assert "ADP value" in recs[0].reason

    def test_combined_reason(self) -> None:
        """Multiple strong secondary factors → combined reason."""
        players = [
            _make_player(1, "Catcher", "C", 15.0),
            # Deep SS for normalization contrast
            _make_player(10, "SS-1", "SS", 30.0),
            _make_player(11, "SS-2", "SS", 12.0),
            _make_player(12, "SS-3", "SS", 8.0),
            _make_player(13, "SS-4", "SS", 4.0),
            _make_player(14, "SS-5", "SS", 2.0),
        ]
        state = _make_state(players, roster_slots={"C": 1, "SS": 1})
        # Both need and scarcity are strong
        w = RecommendationWeights(value=0.5, need=1.0, scarcity=1.0, tier=0.0, adp=0.0)
        recs = recommend(state, weights=w)
        catcher = next(r for r in recs if r.player_id == 1)
        assert "fills need at C" in catcher.reason
        assert "positional scarcity at C" in catcher.reason


# ---------------------------------------------------------------------------
# Step 8: Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_recommendations_change_as_draft_progresses(self) -> None:
        """Recommendations shift after picks are made via DraftEngine."""

        players = [
            _make_player(1, "Best C", "C", 30.0, tier=1),
            _make_player(2, "Top 1B", "1B", 28.0, tier=1),
            _make_player(3, "OF Star", "OF", 25.0, tier=1),
            _make_player(4, "Second C", "C", 15.0, tier=2),
            _make_player(5, "Second 1B", "1B", 14.0, tier=2),
            _make_player(6, "Second OF", "OF", 12.0, tier=2),
            _make_player(7, "Third OF", "OF", 10.0, tier=2),
            _make_player(8, "SP Ace", "SP", 20.0, tier=1, player_type="pitcher"),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"C": 1, "1B": 1, "OF": 2, "P": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        engine = DraftEngine()
        state = engine.start(players, config)

        # Before any picks: best value should be top recommendation
        recs_before = recommend(state, limit=5)
        assert recs_before[0].player_id == 1  # Best C ($30)

        # Team 1 picks Best C
        engine.pick(1, team=1, position="C")
        recs_after_c = recommend(state, limit=5)
        # C no longer in recommendations (C slot filled, no flex)
        ids_after_c = [r.player_id for r in recs_after_c]
        assert 1 not in ids_after_c  # drafted
        assert 4 not in ids_after_c  # C slot full, Second C excluded

        # Team 2 picks Top 1B
        engine.pick(2, team=2, position="1B")

        # Team 2 picks OF Star (round 2, team 2 goes first in snake)
        engine.pick(3, team=2, position="OF")

        # Team 1 picks now — 1B and OF should be boosted
        recs_round2 = recommend(state, limit=5)
        ids_round2 = [r.player_id for r in recs_round2]
        # SP Ace ($20), Second 1B ($14), Second OF ($12), Third OF ($10)
        # SP Ace has highest value but pitcher needs P slot
        assert 8 in ids_round2  # SP Ace still available
        assert 5 in ids_round2  # Second 1B (needed)
        assert 6 in ids_round2  # Second OF (needed)

    def test_mini_draft_simulation(self) -> None:
        """Full mini-draft: 2 teams, 3 slots each, snake format.

        recommend() is only used for user (team 1) picks.
        Opponent (team 2) picks are manual.
        """

        players = [
            _make_player(1, "C Star", "C", 30.0),
            _make_player(2, "1B Star", "1B", 28.0),
            _make_player(3, "OF Star", "OF", 25.0),
            _make_player(4, "C Good", "C", 18.0),
            _make_player(5, "1B Good", "1B", 16.0),
            _make_player(6, "OF Good", "OF", 14.0),
        ]
        config = DraftConfig(
            teams=2,
            roster_slots={"C": 1, "1B": 1, "OF": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        engine = DraftEngine()
        state = engine.start(players, config)

        # Round 1, pick 1: Team 1 picks (use recommendation)
        recs = recommend(state, limit=1)
        assert len(recs) >= 1
        engine.pick(recs[0].player_id, team=1, position=recs[0].position)

        # Round 1, pick 2: Team 2 picks manually
        engine.pick(2, team=2, position="1B")

        # Round 2, pick 3: Team 2 picks first (snake)
        engine.pick(3, team=2, position="OF")

        # Round 2, pick 4: Team 1 picks (use recommendation)
        recs_r2 = recommend(state, limit=1)
        assert len(recs_r2) >= 1
        engine.pick(recs_r2[0].player_id, team=1, position=recs_r2[0].position)

        # Round 3, pick 5: Team 1 picks (use recommendation)
        recs_r3 = recommend(state, limit=1)
        assert len(recs_r3) >= 1
        engine.pick(recs_r3[0].player_id, team=1, position=recs_r3[0].position)

        # User's slots are full → no more recommendations
        final_recs = recommend(state)
        assert final_recs == []

        # Team 1 should have 3 picks
        assert len(state.team_rosters[1]) == 3

    def test_recommendations_reflect_scarcity_shift(self) -> None:
        """As catchers are drafted, scarcity increases for remaining catchers.

        Uses scarcity-heavy weights so the effect is clear.
        SS provides normalization contrast (steep dropoff) so OF stays low.
        """
        players = [
            _make_player(1, "C-1", "C", 25.0),
            _make_player(2, "C-2", "C", 15.0),
            _make_player(3, "OF-1", "OF", 24.0),
            _make_player(4, "OF-2", "OF", 23.5),
            _make_player(5, "OF-3", "OF", 23.0),
            _make_player(6, "OF-4", "OF", 22.5),
            _make_player(7, "OF-5", "OF", 22.0),
            # SS with steep dropoff for normalization contrast
            _make_player(10, "SS-1", "SS", 25.0),
            _make_player(11, "SS-2", "SS", 8.0),
            _make_player(12, "SS-3", "SS", 5.0),
            _make_player(13, "SS-4", "SS", 3.0),
            _make_player(14, "SS-5", "SS", 1.0),
        ]
        w = RecommendationWeights(value=0.5, need=0.0, scarcity=1.0, tier=0.0, adp=0.0)
        state = _make_state(players, roster_slots={"C": 1, "OF": 3, "SS": 1}, teams=2)

        recs_initial = recommend(state, weights=w)
        c1_initial = next(r for r in recs_initial if r.player_id == 1)

        # Opponent drafts C-1
        pick = DraftPick(pick_number=1, team=2, player_id=1, player_name="C-1", position="C")
        state.picks.append(pick)
        state.team_rosters[2].append(pick)
        del state.available_pool[(1, "batter")]
        state.current_pick = 2

        # Now C-2 is the only catcher — maximum scarcity
        recs_after = recommend(state, weights=w)
        c2 = next(r for r in recs_after if r.player_id == 2)
        # C-2's scarcity score should be higher than C-1's was initially
        # (C-1 had a backup in C-2, but C-2 has none)
        assert c2.score > c1_initial.score


# ---------------------------------------------------------------------------
# Step 9: Category balance scoring
# ---------------------------------------------------------------------------


class TestCategoryBalance:
    def test_category_balance_fn_boosts_weak_category_player(self) -> None:
        """Player addressing a weak category gets boosted by category_balance_fn."""
        players = [
            _make_player(1, "SB Guy", "OF", 18.0),
            _make_player(2, "HR Guy", "OF", 20.0),
        ]
        state = _make_state(players, roster_slots={"OF": 3})

        # SB Guy addresses weak categories, HR Guy doesn't
        def cat_balance_fn(roster_ids: list[int], available_ids: list[int]) -> dict[int, float]:
            return {1: 1.0, 2: 0.0}

        w = RecommendationWeights(value=1.0, need=0.0, scarcity=0.0, tier=0.0, adp=0.0, category_balance=2.0)
        recs = recommend(state, weights=w, category_balance_fn=cat_balance_fn)
        # SB Guy should outrank HR Guy despite lower value
        assert recs[0].player_id == 1

    def test_category_balance_fn_none_backward_compatible(self) -> None:
        """category_balance_fn=None → no effect, backward compatible."""
        players = [
            _make_player(1, "Player A", "OF", 25.0),
            _make_player(2, "Player B", "OF", 20.0),
        ]
        state = _make_state(players, roster_slots={"OF": 2})
        w = RecommendationWeights(value=1.0, need=0.0, scarcity=0.0, tier=0.0, adp=0.0)
        recs = recommend(state, weights=w)
        assert recs[0].player_id == 1

    def test_cat_scores_param_bypasses_category_balance_fn(self) -> None:
        """Pre-computed cat_scores dict is used directly, skipping category_balance_fn."""
        players = [
            _make_player(1, "SB Guy", "OF", 18.0),
            _make_player(2, "HR Guy", "OF", 20.0),
        ]
        state = _make_state(players, roster_slots={"OF": 3})

        w = RecommendationWeights(value=1.0, need=0.0, scarcity=0.0, tier=0.0, adp=0.0, category_balance=2.0)
        recs = recommend(state, weights=w, cat_scores={1: 1.0, 2: 0.0})
        # SB Guy should outrank HR Guy despite lower value
        assert recs[0].player_id == 1

    def test_cat_scores_takes_precedence_over_category_balance_fn(self) -> None:
        """When both cat_scores and category_balance_fn are provided, cat_scores wins."""
        players = [
            _make_player(1, "Player A", "OF", 18.0),
            _make_player(2, "Player B", "OF", 20.0),
        ]
        state = _make_state(players, roster_slots={"OF": 3})

        def cat_balance_fn(roster_ids: list[int], available_ids: list[int]) -> dict[int, float]:
            return {1: 1.0, 2: 0.0}

        w = RecommendationWeights(value=1.0, need=0.0, scarcity=0.0, tier=0.0, adp=0.0, category_balance=2.0)
        recs = recommend(state, weights=w, category_balance_fn=cat_balance_fn, cat_scores={1: 0.0, 2: 1.0})
        # Player B should win because cat_scores takes precedence
        assert recs[0].player_id == 2

    def test_reason_includes_category_balance(self) -> None:
        """Reason includes 'addresses weak categories' when cat_bal > 0.3."""
        players = [
            _make_player(1, "Balance Guy", "OF", 15.0),
        ]
        state = _make_state(players, roster_slots={"OF": 2})

        def cat_balance_fn(roster_ids: list[int], available_ids: list[int]) -> dict[int, float]:
            return {1: 0.8}

        w = RecommendationWeights(value=0.5, need=0.0, scarcity=0.0, tier=0.0, adp=0.0, category_balance=1.0)
        recs = recommend(state, weights=w, category_balance_fn=cat_balance_fn)
        assert "addresses weak categories" in recs[0].reason

    def test_reason_includes_specific_weak_categories(self) -> None:
        """Reason lists specific weak categories when weak_categories provided."""
        players = [
            _make_player(1, "Balance Guy", "OF", 15.0),
        ]
        state = _make_state(players, roster_slots={"OF": 2})

        def cat_balance_fn(roster_ids: list[int], available_ids: list[int]) -> dict[int, float]:
            return {1: 0.8}

        w = RecommendationWeights(value=0.5, need=0.0, scarcity=0.0, tier=0.0, adp=0.0, category_balance=1.0)
        recs = recommend(state, weights=w, category_balance_fn=cat_balance_fn, weak_categories=["SB", "ERA"])
        assert "fills SB + ERA gaps" in recs[0].reason

    def test_weak_categories_none_uses_generic_message(self) -> None:
        """weak_categories=None falls back to generic 'addresses weak categories'."""
        players = [
            _make_player(1, "Balance Guy", "OF", 15.0),
        ]
        state = _make_state(players, roster_slots={"OF": 2})

        def cat_balance_fn(roster_ids: list[int], available_ids: list[int]) -> dict[int, float]:
            return {1: 0.8}

        w = RecommendationWeights(value=0.5, need=0.0, scarcity=0.0, tier=0.0, adp=0.0, category_balance=1.0)
        recs = recommend(state, weights=w, category_balance_fn=cat_balance_fn, weak_categories=None)
        assert "addresses weak categories" in recs[0].reason


# ---------------------------------------------------------------------------
# Step 10: Mock position bonus
# ---------------------------------------------------------------------------


def _make_plan(targets: list[DraftPlanTarget]) -> DraftPlan:
    return DraftPlan(
        slot=1,
        teams=4,
        strategy_name="best-value",
        targets=targets,
        n_simulations=100,
        avg_roster_value=50.0,
    )


def _make_availability(
    player_id: int, available_at: float, *, name: str = "Player", position: str = "OF"
) -> AvailabilityWindow:
    return AvailabilityWindow(
        player_id=player_id,
        player_name=name,
        position=position,
        earliest_pick=1.0,
        median_pick=10.0,
        latest_pick=20.0,
        available_at_user_pick=available_at,
    )


class TestMockPositionBonus:
    def test_position_matches_plan_target(self) -> None:
        """Player at position matching round target gets boosted vs non-target."""
        plan = _make_plan([DraftPlanTarget(round_range=(1, 2), position="C", confidence=0.8, example_players=[])])
        players = [
            _make_player(1, "Catcher", "C", 20.0),
            _make_player(2, "First Base", "1B", 20.0),
        ]
        state = _make_state(players, roster_slots={"C": 1, "1B": 1})
        w = RecommendationWeights(
            value=1.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=1.0,
            mock_availability=0.0,
        )
        recs = recommend(state, weights=w, draft_plan=plan)
        catcher = next(r for r in recs if r.player_id == 1)
        first_base = next(r for r in recs if r.player_id == 2)
        assert catcher.score > first_base.score

    def test_confidence_scales_bonus(self) -> None:
        """High confidence target produces higher bonus than low confidence."""
        plan_high = _make_plan([DraftPlanTarget(round_range=(1, 2), position="C", confidence=0.9, example_players=[])])
        plan_low = _make_plan([DraftPlanTarget(round_range=(1, 2), position="C", confidence=0.3, example_players=[])])
        players = [_make_player(1, "Catcher", "C", 20.0)]
        w = RecommendationWeights(
            value=0.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=1.0,
            mock_availability=0.0,
        )

        state_high = _make_state(players, roster_slots={"C": 1})
        recs_high = recommend(state_high, weights=w, draft_plan=plan_high)

        state_low = _make_state(players, roster_slots={"C": 1})
        recs_low = recommend(state_low, weights=w, draft_plan=plan_low)

        assert recs_high[0].score > recs_low[0].score

    def test_no_plan_no_effect(self) -> None:
        """draft_plan=None produces identical scores to baseline."""
        players = [_make_player(1, "Catcher", "C", 20.0)]
        state = _make_state(players, roster_slots={"C": 1})
        w = RecommendationWeights(
            value=1.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=1.0,
            mock_availability=0.0,
        )
        recs_with = recommend(state, weights=w, draft_plan=None)
        recs_without = recommend(state, weights=w)
        assert recs_with[0].score == recs_without[0].score

    def test_round_outside_target_range(self) -> None:
        """Round outside any target range returns 0.0 bonus."""
        # Target is rounds 5-6, but we're in round 1
        plan = _make_plan([DraftPlanTarget(round_range=(5, 6), position="C", confidence=0.8, example_players=[])])
        players = [
            _make_player(1, "Catcher", "C", 20.0),
            _make_player(2, "First Base", "1B", 20.0),
        ]
        state = _make_state(players, roster_slots={"C": 1, "1B": 1})
        w = RecommendationWeights(
            value=0.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=1.0,
            mock_availability=0.0,
        )
        recs = recommend(state, weights=w, draft_plan=plan)
        # Both should score 0 — no bonus for either
        assert recs[0].score == 0.0


# ---------------------------------------------------------------------------
# Step 11: Mock availability
# ---------------------------------------------------------------------------


class TestMockAvailability:
    def test_high_availability_wait_penalty(self) -> None:
        """Player with ≥0.8 availability scores lower than one with 0.2 availability."""
        avail = [
            _make_availability(1, 0.9, name="Safe Player", position="C"),
            _make_availability(2, 0.2, name="Scarce Player", position="1B"),
        ]
        players = [
            _make_player(1, "Safe Player", "C", 20.0),
            _make_player(2, "Scarce Player", "1B", 20.0),
        ]
        # User is team 1, pick 2 (not on clock), so picks_until > 0
        state = _make_state(players, roster_slots={"C": 1, "1B": 1}, teams=4, user_team=1)
        state.current_pick = 2  # not user's pick
        w = RecommendationWeights(
            value=1.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=0.0,
            mock_availability=1.0,
        )
        recs = recommend(state, weights=w, availability=avail)
        safe = next(r for r in recs if r.player_id == 1)
        scarce = next(r for r in recs if r.player_id == 2)
        assert scarce.score > safe.score

    def test_low_availability_urgency_bonus(self) -> None:
        """Player about to be taken (<0.3 availability) gets urgency bonus."""
        avail = [_make_availability(1, 0.1, name="Hot Player", position="C")]
        players = [_make_player(1, "Hot Player", "C", 20.0)]
        state = _make_state(players, roster_slots={"C": 1}, teams=4, user_team=1)
        state.current_pick = 2
        w = RecommendationWeights(
            value=0.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=0.0,
            mock_availability=1.0,
        )
        recs = recommend(state, weights=w, availability=avail)
        assert recs[0].score > 0

    def test_no_availability_no_effect(self) -> None:
        """availability=None identical to baseline."""
        players = [_make_player(1, "Player", "C", 20.0)]
        state = _make_state(players, roster_slots={"C": 1})
        w = RecommendationWeights(
            value=1.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=0.0,
            mock_availability=1.0,
        )
        recs_with = recommend(state, weights=w, availability=None)
        recs_without = recommend(state, weights=w)
        assert recs_with[0].score == recs_without[0].score

    def test_user_on_clock_no_availability_signal(self) -> None:
        """When picks_until=0 (user on clock), availability doesn't matter."""
        avail = [
            _make_availability(1, 0.9, name="Safe", position="C"),
            _make_availability(2, 0.1, name="Hot", position="1B"),
        ]
        players = [
            _make_player(1, "Safe", "C", 20.0),
            _make_player(2, "Hot", "1B", 20.0),
        ]
        # User is on the clock (current_pick=1, team 1)
        state = _make_state(players, roster_slots={"C": 1, "1B": 1}, teams=4, user_team=1)
        w = RecommendationWeights(
            value=0.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=0.0,
            mock_availability=1.0,
        )
        recs = recommend(state, weights=w, availability=avail)
        # Both should score 0 — no availability signal when on the clock
        for r in recs:
            assert r.score == 0.0


# ---------------------------------------------------------------------------
# Step 12: Mock plan integration
# ---------------------------------------------------------------------------


class TestMockPlanIntegration:
    def test_combined_plan_and_availability(self) -> None:
        """Plan-targeted position with low availability outranks non-targeted with high availability."""
        plan = _make_plan([DraftPlanTarget(round_range=(1, 2), position="C", confidence=0.8, example_players=[])])
        avail = [
            _make_availability(1, 0.15, name="Catcher", position="C"),
            _make_availability(2, 0.9, name="First Base", position="1B"),
        ]
        players = [
            _make_player(1, "Catcher", "C", 18.0),
            _make_player(2, "First Base", "1B", 20.0),
        ]
        state = _make_state(players, roster_slots={"C": 1, "1B": 1}, teams=4, user_team=1)
        state.current_pick = 2  # not on clock
        w = RecommendationWeights(
            value=1.0,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=1.0,
            mock_availability=1.0,
        )
        recs = recommend(state, weights=w, draft_plan=plan, availability=avail)
        assert recs[0].player_id == 1

    def test_reason_includes_mock_signals(self) -> None:
        """Reason strings include mock plan and availability signals."""
        plan = _make_plan([DraftPlanTarget(round_range=(1, 2), position="C", confidence=0.8, example_players=[])])
        avail = [_make_availability(1, 0.1, name="Catcher", position="C")]
        players = [_make_player(1, "Catcher", "C", 20.0)]
        state = _make_state(players, roster_slots={"C": 1}, teams=4, user_team=1)
        state.current_pick = 2
        w = RecommendationWeights(
            value=0.5,
            need=0.0,
            scarcity=0.0,
            tier=0.0,
            adp=0.0,
            category_balance=0.0,
            mock_position=1.0,
            mock_availability=1.0,
        )
        recs = recommend(state, weights=w, draft_plan=plan, availability=avail)
        assert "mock plan targets C" in recs[0].reason
        assert "mock sims: likely gone" in recs[0].reason
