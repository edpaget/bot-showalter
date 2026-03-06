from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.tier import PlayerTier
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.roster_optimizer import optimize_auction_budget


def _league(
    budget: int = 260,
    positions: dict[str, int] | None = None,
    pitcher_positions: dict[str, int] | None = None,
    roster_util: int = 0,
) -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=budget,
        roster_batters=9,
        roster_pitchers=8,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="W", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=roster_util,
        positions=positions or {"c": 1, "1b": 1, "ss": 1, "of": 3},
        pitcher_positions=pitcher_positions or {"sp": 2, "rp": 2},
    )


def _valuation(player_id: int, position: str, value: float, player_type: str = "batter") -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2025,
        system="zar",
        version="1.0",
        projection_system="steamer",
        projection_version="1.0",
        player_type=player_type,
        position=position,
        value=value,
        rank=1,
        category_scores={},
    )


def _make_valuations() -> tuple[list[Valuation], dict[int, str]]:
    """Build a set of valuations with known values for testing."""
    valuations: list[Valuation] = []
    names: dict[int, str] = {}
    pid = 1

    # Catchers: 3 players, descending values
    for val, name in [(30.0, "Catcher A"), (15.0, "Catcher B"), (5.0, "Catcher C")]:
        valuations.append(_valuation(pid, "c", val))
        names[pid] = name
        pid += 1

    # 1B: 3 players
    for val, name in [(25.0, "First A"), (20.0, "First B"), (8.0, "First C")]:
        valuations.append(_valuation(pid, "1b", val))
        names[pid] = name
        pid += 1

    # SS: 3 players
    for val, name in [(35.0, "Short A"), (18.0, "Short B"), (6.0, "Short C")]:
        valuations.append(_valuation(pid, "ss", val))
        names[pid] = name
        pid += 1

    # OF: 6 players (need 3 slots)
    for val, name in [
        (40.0, "OF A"),
        (32.0, "OF B"),
        (28.0, "OF C"),
        (15.0, "OF D"),
        (10.0, "OF E"),
        (3.0, "OF F"),
    ]:
        valuations.append(_valuation(pid, "of", val))
        names[pid] = name
        pid += 1

    # SP: 5 players
    for val, name in [
        (38.0, "SP A"),
        (30.0, "SP B"),
        (20.0, "SP C"),
        (12.0, "SP D"),
        (5.0, "SP E"),
    ]:
        valuations.append(_valuation(pid, "sp", val, player_type="pitcher"))
        names[pid] = name
        pid += 1

    # RP: 4 players
    for val, name in [(22.0, "RP A"), (14.0, "RP B"), (7.0, "RP C"), (2.0, "RP D")]:
        valuations.append(_valuation(pid, "rp", val, player_type="pitcher"))
        names[pid] = name
        pid += 1

    return valuations, names


class TestBudgetSumsToLeagueBudget:
    def test_balanced(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="balanced")
        total = sum(a.budget for a in result)
        assert total == league.budget

    def test_stars_and_scrubs(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="stars_and_scrubs")
        total = sum(a.budget for a in result)
        assert total == league.budget


class TestAllRosterSlotsAccountedFor:
    def test_slot_count(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names)
        # c=1, 1b=1, ss=1, of=3, sp=2, rp=2 = 10
        assert len(result) == 10

    def test_with_util(self) -> None:
        league = _league(roster_util=1)
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names)
        assert len(result) == 11


class TestMinimumBidFloor:
    def test_every_allocation_at_least_one(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        for strategy in ("balanced", "stars_and_scrubs"):
            result = optimize_auction_budget(valuations, league, names, strategy=strategy)
            for alloc in result:
                assert alloc.budget >= 1.0, f"{alloc.position} budget {alloc.budget} < 1 ({strategy})"


class TestStarsAndScrubsConcentrates:
    def test_top_5_over_half(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="stars_and_scrubs")
        sorted_allocs = sorted(result, key=lambda a: a.budget, reverse=True)
        top_5 = sum(a.budget for a in sorted_allocs[:5])
        assert top_5 > league.budget * 0.5


class TestBalancedWithin2x:
    def test_non_minimum_within_2x(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="balanced")
        non_min = [a for a in result if a.budget > 1.0]
        if len(non_min) >= 2:
            budgets = [a.budget for a in non_min]
            assert max(budgets) <= 2 * min(budgets)


class TestEdgeCaseBudgetEqualsSlots:
    def test_all_get_one(self) -> None:
        league = _league(budget=10)  # exactly 10 slots
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="balanced")
        for alloc in result:
            assert alloc.budget == 1.0

    def test_stars_all_get_one(self) -> None:
        league = _league(budget=10)
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="stars_and_scrubs")
        for alloc in result:
            assert alloc.budget == 1.0


class TestTiersPopulated:
    def test_tiers_set_when_provided(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        tiers = [
            PlayerTier(player_id=1, player_name="Catcher A", position="c", tier=1, value=10.0, rank=1),
            PlayerTier(player_id=2, player_name="Catcher B", position="c", tier=2, value=5.0, rank=2),
            PlayerTier(player_id=7, player_name="Short A", position="ss", tier=1, value=10.0, rank=1),
        ]
        result = optimize_auction_budget(valuations, league, names, tiers=tiers)
        tier_set = [a for a in result if a.target_tier is not None]
        assert len(tier_set) > 0

    def test_no_tiers_when_not_provided(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names)
        for alloc in result:
            assert alloc.target_tier is None


class TestTargetPlayerNames:
    def test_names_populated(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="balanced")
        for alloc in result:
            assert len(alloc.target_player_names) > 0
