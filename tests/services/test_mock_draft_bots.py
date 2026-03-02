import random

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
    PositionalNeedBot,
    RandomBot,
)


def _make_row(
    player_id: int,
    name: str,
    position: str,
    value: float,
    *,
    adp: float | None = None,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type="B" if position != "SP" else "P",
        position=position,
        value=value,
        category_z_scores={},
        adp_overall=adp,
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
