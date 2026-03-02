import dataclasses
import random

import pytest

from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.mock_draft import DraftPick
from fantasy_baseball_manager.services.mock_draft_bots import (
    ADPBot,
    BestValueBot,
    CategoryNeedRule,
    CompositeBot,
    FallbackBestValueRule,
    PositionalNeedBot,
    PositionTargetRule,
    RandomBot,
    TierValueRule,
    WeightedRule,
)


def _make_row(
    player_id: int,
    name: str,
    position: str,
    value: float,
    *,
    adp: float | None = None,
    tier: int | None = None,
    category_z_scores: dict[str, float] | None = None,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type="B" if position != "SP" else "P",
        position=position,
        value=value,
        category_z_scores=category_z_scores or {},
        adp_overall=adp,
        tier=tier,
    )


def _make_league() -> LeagueSettings:
    batting_cat = CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
    pitching_cat = CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
    return LeagueSettings(
        name="Test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=4,
        budget=260,
        roster_batters=3,
        roster_pitchers=2,
        batting_categories=(batting_cat,),
        pitching_categories=(pitching_cat,),
        positions={"C": 1, "1B": 1, "OF": 1},
    )


AVAILABLE = [
    _make_row(1, "Star C", "C", 30.0, adp=5.0),
    _make_row(2, "Great 1B", "1B", 25.0, adp=10.0),
    _make_row(3, "Good OF", "OF", 20.0, adp=15.0),
    _make_row(4, "OK SP", "SP", 18.0, adp=20.0),
    _make_row(5, "Avg C", "C", 12.0, adp=30.0),
    _make_row(6, "No ADP", "OF", 8.0, adp=None),
]


class TestADPBot:
    def test_picks_lowest_adp(self) -> None:
        bot = ADPBot(rng=random.Random(42))
        league = _make_league()
        player_id = bot.pick(AVAILABLE, [], league)
        assert player_id == 1  # ADP 5.0 is lowest

    def test_none_adp_picked_last(self) -> None:
        # Only players with None ADP and one with ADP
        available = [
            _make_row(6, "No ADP", "OF", 8.0, adp=None),
            _make_row(4, "Has ADP", "SP", 18.0, adp=100.0),
        ]
        bot = ADPBot(rng=random.Random(42))
        league = _make_league()
        player_id = bot.pick(available, [], league)
        assert player_id == 4  # Has ADP 100 < infinity


class TestBestValueBot:
    def test_picks_highest_value(self) -> None:
        bot = BestValueBot(rng=random.Random(42))
        league = _make_league()
        player_id = bot.pick(AVAILABLE, [], league)
        assert player_id == 1  # value 30.0 is highest


class TestPositionalNeedBot:
    def test_prefers_needed_position(self) -> None:
        bot = PositionalNeedBot(rng=random.Random(42))
        league = _make_league()
        # Already have a C — should prefer other positions
        existing_roster = [
            DraftPick(round=1, pick=1, team_idx=0, player_id=1, player_name="Star C", position="C", value=30.0),
        ]
        # Available: C(12.0), 1B(25.0), OF(20.0), SP(18.0)
        available = [
            _make_row(5, "Avg C", "C", 12.0, adp=30.0),
            _make_row(2, "Great 1B", "1B", 25.0, adp=10.0),
            _make_row(3, "Good OF", "OF", 20.0, adp=15.0),
            _make_row(4, "OK SP", "SP", 18.0, adp=20.0),
        ]
        player_id = bot.pick(available, existing_roster, league)
        # Should pick 1B (highest value among needed positions)
        assert player_id == 2

    def test_falls_back_to_highest_value(self) -> None:
        bot = PositionalNeedBot(rng=random.Random(42))
        league = _make_league()
        # All primary positions filled — falls back to best value
        existing_roster = [
            DraftPick(round=1, pick=1, team_idx=0, player_id=1, player_name="C", position="C", value=30.0),
            DraftPick(round=2, pick=2, team_idx=0, player_id=2, player_name="1B", position="1B", value=25.0),
            DraftPick(round=3, pick=3, team_idx=0, player_id=3, player_name="OF", position="OF", value=20.0),
        ]
        available = [
            _make_row(5, "Avg C", "C", 12.0, adp=30.0),
            _make_row(6, "Extra OF", "OF", 8.0, adp=None),
        ]
        player_id = bot.pick(available, existing_roster, league)
        # Falls back to highest value
        assert player_id == 5


class TestRandomBot:
    def test_picks_from_top_20(self) -> None:
        bot = RandomBot(rng=random.Random(42))
        league = _make_league()
        player_id = bot.pick(AVAILABLE, [], league)
        assert player_id in {row.player_id for row in AVAILABLE}

    def test_deterministic_with_same_seed(self) -> None:
        bot1 = RandomBot(rng=random.Random(42))
        bot2 = RandomBot(rng=random.Random(42))
        league = _make_league()
        pick1 = bot1.pick(AVAILABLE, [], league)
        pick2 = bot2.pick(AVAILABLE, [], league)
        assert pick1 == pick2

    def test_different_seeds_may_differ(self) -> None:
        league = _make_league()
        # Run enough times to verify different seeds can produce different picks
        picks: set[int] = set()
        for seed in range(100):
            bot = RandomBot(rng=random.Random(seed))
            picks.add(bot.pick(AVAILABLE, [], league))
        # With 6 available players and 100 seeds, should get more than 1 unique pick
        assert len(picks) > 1


# ---------------------------------------------------------------------------
# WeightedRule tests
# ---------------------------------------------------------------------------


class TestWeightedRule:
    def test_frozen(self) -> None:
        rule = FallbackBestValueRule()
        wr = WeightedRule(rule=rule, weight=2.0)
        assert wr.rule is rule
        assert wr.weight == 2.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            wr.weight = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TierValueRule tests
# ---------------------------------------------------------------------------


class TestTierValueRule:
    def test_tier_1_doubles_value(self) -> None:
        rule = TierValueRule()
        player = _make_row(1, "T1", "C", 15.0, tier=1)
        assert rule.score(player, [], 1, _make_league()) == 30.0

    def test_tier_2_passes_value(self) -> None:
        rule = TierValueRule()
        player = _make_row(1, "T2", "C", 15.0, tier=2)
        assert rule.score(player, [], 1, _make_league()) == 15.0

    def test_tier_3_returns_none(self) -> None:
        rule = TierValueRule()
        player = _make_row(1, "T3", "C", 15.0, tier=3)
        assert rule.score(player, [], 1, _make_league()) is None

    def test_no_tier_returns_none(self) -> None:
        rule = TierValueRule()
        player = _make_row(1, "NoTier", "C", 15.0, tier=None)
        assert rule.score(player, [], 1, _make_league()) is None


# ---------------------------------------------------------------------------
# PositionTargetRule tests
# ---------------------------------------------------------------------------


class TestPositionTargetRule:
    def test_matching_position_and_round(self) -> None:
        rule = PositionTargetRule(position="C", rounds=(3, 5))
        player = _make_row(1, "Catcher", "C", 10.0)
        assert rule.score(player, [], 4, _make_league()) == 10.0

    def test_matching_position_round_start(self) -> None:
        rule = PositionTargetRule(position="C", rounds=(3, 5))
        player = _make_row(1, "Catcher", "C", 10.0)
        assert rule.score(player, [], 3, _make_league()) == 10.0

    def test_matching_position_round_end(self) -> None:
        rule = PositionTargetRule(position="C", rounds=(3, 5))
        player = _make_row(1, "Catcher", "C", 10.0)
        assert rule.score(player, [], 5, _make_league()) == 10.0

    def test_wrong_position_returns_none(self) -> None:
        rule = PositionTargetRule(position="C", rounds=(3, 5))
        player = _make_row(1, "First", "1B", 10.0)
        assert rule.score(player, [], 4, _make_league()) is None

    def test_wrong_round_returns_none(self) -> None:
        rule = PositionTargetRule(position="C", rounds=(3, 5))
        player = _make_row(1, "Catcher", "C", 10.0)
        assert rule.score(player, [], 2, _make_league()) is None


# ---------------------------------------------------------------------------
# CategoryNeedRule tests
# ---------------------------------------------------------------------------


class TestCategoryNeedRule:
    def test_empty_roster_returns_none(self) -> None:
        rule = CategoryNeedRule(z_score_lookup={})
        player = _make_row(1, "P", "C", 10.0, category_z_scores={"HR": 2.0, "SB": 1.0})
        assert rule.score(player, [], 1, _make_league()) is None

    def test_boosts_weakest_category(self) -> None:
        # Roster has player 10 who is strong in HR but weak in SB
        lookup: dict[int, dict[str, float]] = {
            10: {"HR": 3.0, "SB": 0.5},
        }
        rule = CategoryNeedRule(z_score_lookup=lookup)
        roster = [DraftPick(round=1, pick=1, team_idx=0, player_id=10, player_name="X", position="C", value=20.0)]
        # Player strong in SB (the weak category)
        player_sb = _make_row(1, "SB Guy", "OF", 10.0, category_z_scores={"HR": 0.0, "SB": 2.5})
        score = rule.score(player_sb, roster, 2, _make_league())
        assert score == 2.5  # z-score in weakest category (SB)

    def test_negative_zscore_returns_none(self) -> None:
        lookup: dict[int, dict[str, float]] = {
            10: {"HR": 3.0, "SB": 0.5},
        }
        rule = CategoryNeedRule(z_score_lookup=lookup)
        roster = [DraftPick(round=1, pick=1, team_idx=0, player_id=10, player_name="X", position="C", value=20.0)]
        # Player has negative z in weakest category
        player = _make_row(1, "Bad SB", "OF", 10.0, category_z_scores={"HR": 1.0, "SB": -0.5})
        assert rule.score(player, roster, 2, _make_league()) is None

    def test_roster_player_not_in_lookup_ignored(self) -> None:
        lookup: dict[int, dict[str, float]] = {
            10: {"HR": 3.0, "SB": 0.5},
        }
        rule = CategoryNeedRule(z_score_lookup=lookup)
        roster = [
            DraftPick(round=1, pick=1, team_idx=0, player_id=10, player_name="X", position="C", value=20.0),
            DraftPick(round=2, pick=2, team_idx=0, player_id=99, player_name="Unknown", position="1B", value=15.0),
        ]
        player = _make_row(1, "SB Guy", "OF", 10.0, category_z_scores={"HR": 0.0, "SB": 2.0})
        # Should still work — player 99 not in lookup, just ignored
        score = rule.score(player, roster, 3, _make_league())
        assert score == 2.0


# ---------------------------------------------------------------------------
# FallbackBestValueRule tests
# ---------------------------------------------------------------------------


class TestFallbackBestValueRule:
    def test_returns_value(self) -> None:
        rule = FallbackBestValueRule()
        player = _make_row(1, "P", "C", 17.5)
        assert rule.score(player, [], 1, _make_league()) == 17.5

    def test_never_returns_none(self) -> None:
        rule = FallbackBestValueRule()
        player = _make_row(1, "P", "C", 0.0)
        result = rule.score(player, [], 1, _make_league())
        assert result is not None
        assert result == 0.0


# ---------------------------------------------------------------------------
# CompositeBot tests
# ---------------------------------------------------------------------------


class TestCompositeBot:
    def test_single_fallback_picks_highest_value(self) -> None:
        """Single FallbackBestValueRule → picks highest value (same as BestValueBot)."""
        rules = [WeightedRule(rule=FallbackBestValueRule(), weight=1.0)]
        bot = CompositeBot(rules=rules, rng=random.Random(42))
        league = _make_league()
        player_id = bot.pick(AVAILABLE, [], league)
        assert player_id == 1  # value 30.0 is highest

    def test_tier_value_overrides_raw_value(self) -> None:
        """TierValueRule(weight=10) + FallbackBestValueRule(weight=1) → prefers tier-1."""
        # Tier 1 player has lower raw value but tier boost makes it win
        tier1 = _make_row(10, "Tier1", "C", 15.0, tier=1)
        non_tier = _make_row(11, "NoTier", "1B", 25.0, tier=None)
        available = [non_tier, tier1]

        rules = [
            WeightedRule(rule=TierValueRule(), weight=10.0),
            WeightedRule(rule=FallbackBestValueRule(), weight=1.0),
        ]
        bot = CompositeBot(rules=rules, rng=random.Random(42))
        league = _make_league()
        # tier1 score: 10*(15*2) + 1*15 = 315; non_tier: 10*None + 1*25 = 25
        player_id = bot.pick(available, [], league)
        assert player_id == 10

    def test_all_rules_return_none_falls_back(self) -> None:
        """All rules return None → falls back to first available."""
        # TierValueRule only scores tier 1-2. If no tiers, all return None.
        available = [
            _make_row(1, "A", "C", 10.0, tier=None),
            _make_row(2, "B", "1B", 20.0, tier=None),
        ]
        rules = [WeightedRule(rule=TierValueRule(), weight=1.0)]
        bot = CompositeBot(rules=rules, rng=random.Random(42))
        league = _make_league()
        player_id = bot.pick(available, [], league)
        assert player_id == 1  # first available
