import dataclasses

import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.valuation_evaluator import ValuationEvaluator
from tests.fakes.repos import (
    FakeBattingStatsRepo,
    FakePlayerRepo,
    FakePitchingStatsRepo,
    FakePositionAppearanceRepo,
    FakeValuationRepo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Simple counting-only league: 2 teams, HR + R for batters, W + SV for pitchers.
# Roster: 2 batter slots (1 of + 1 util) + 1 pitcher slot.


def _counting_league() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=260,
        roster_batters=2,
        roster_pitchers=1,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=1,
        positions={"of": 1},
    )


def _standard_players() -> list[Player]:
    return [
        Player(id=1, name_first="Mike", name_last="Trout"),
        Player(id=2, name_first="Aaron", name_last="Judge"),
        Player(id=3, name_first="Mookie", name_last="Betts"),
        Player(id=4, name_first="Gerrit", name_last="Cole"),
        Player(id=5, name_first="Edwin", name_last="Diaz"),
    ]


def _standard_valuations() -> list[Valuation]:
    """Predicted valuations for 3 batters + 2 pitchers."""
    return [
        Valuation(
            player_id=1,
            season=2025,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="batter",
            position="of",
            value=40.0,
            rank=1,
            category_scores={"hr": 1.5, "r": 1.0},
        ),
        Valuation(
            player_id=2,
            season=2025,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="batter",
            position="util",
            value=30.0,
            rank=2,
            category_scores={"hr": 0.8, "r": 0.5},
        ),
        Valuation(
            player_id=3,
            season=2025,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="batter",
            position="of",
            value=20.0,
            rank=3,
            category_scores={"hr": 0.0, "r": 0.2},
        ),
        Valuation(
            player_id=4,
            season=2025,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="pitcher",
            position="p",
            value=25.0,
            rank=4,
            category_scores={"w": 1.0, "sv": -0.5},
        ),
        Valuation(
            player_id=5,
            season=2025,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="pitcher",
            position="p",
            value=15.0,
            rank=5,
            category_scores={"w": -0.5, "sv": 1.5},
        ),
    ]


def _standard_batting_actuals() -> list[BattingStats]:
    return [
        BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=35, r=100),
        BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=45, r=110),
        BattingStats(player_id=3, season=2025, source="fangraphs", pa=500, hr=20, r=80),
    ]


def _standard_pitching_actuals() -> list[PitchingStats]:
    return [
        PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0),
        PitchingStats(player_id=5, season=2025, source="fangraphs", ip=70.0, w=5, sv=30),
    ]


def _standard_appearances() -> list[PositionAppearance]:
    return [
        PositionAppearance(player_id=1, season=2025, position="OF", games=150),
        PositionAppearance(player_id=2, season=2025, position="OF", games=140),
        PositionAppearance(player_id=3, season=2025, position="OF", games=130),
    ]


def _build_evaluator(
    valuations: list[Valuation] | None = None,
    batting: list[BattingStats] | None = None,
    pitching: list[PitchingStats] | None = None,
    appearances: list[PositionAppearance] | None = None,
    players: list[Player] | None = None,
) -> ValuationEvaluator:
    return ValuationEvaluator(
        valuation_repo=FakeValuationRepo(valuations if valuations is not None else _standard_valuations()),
        batting_repo=FakeBattingStatsRepo(batting if batting is not None else _standard_batting_actuals()),
        pitching_repo=FakePitchingStatsRepo(pitching if pitching is not None else _standard_pitching_actuals()),
        position_repo=FakePositionAppearanceRepo(appearances if appearances is not None else _standard_appearances()),
        player_repo=FakePlayerRepo(players if players is not None else _standard_players()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValuationEvaluator:
    def test_evaluate_returns_result(self) -> None:
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.system == "zar"
        assert result.version == "1.0"
        assert result.season == 2025
        assert result.n == 5

    def test_value_mae_computed(self) -> None:
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.value_mae >= 0.0
        # MAE should be finite and meaningful
        assert result.value_mae < 1000.0

    def test_rank_correlation_computed(self) -> None:
        """With enough players, correlation should be defined."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert -1.0 <= result.rank_correlation <= 1.0

    def test_rank_correlation_inverted(self) -> None:
        """If predicted rankings are opposite to actual, correlation should be negative."""
        # Create valuations where the worst predicted players actually did best
        valuations = [
            Valuation(
                player_id=1,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=10.0,
                rank=3,
                category_scores={"hr": -1.0, "r": -1.0},
            ),
            Valuation(
                player_id=2,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="util",
                value=20.0,
                rank=2,
                category_scores={"hr": 0.0, "r": 0.0},
            ),
            Valuation(
                player_id=3,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=30.0,
                rank=1,
                category_scores={"hr": 1.0, "r": 1.0},
            ),
        ]
        # Actuals: player 1 best, player 3 worst (inverted)
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=50, r=120),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=25, r=80),
            BattingStats(player_id=3, season=2025, source="fangraphs", pa=500, hr=5, r=40),
        ]
        evaluator = _build_evaluator(valuations=valuations, batting=batting, pitching=[])
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.rank_correlation < 0.0

    def test_players_without_actuals_excluded(self) -> None:
        """A player with a valuation but no actual stats should be excluded."""
        # Player 3 has valuation but no actuals
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=35, r=100),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=45, r=110),
        ]
        evaluator = _build_evaluator(batting=batting)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        player_ids = {p.player_id for p in result.players}
        assert 3 not in player_ids

    def test_players_without_valuations_excluded(self) -> None:
        """A player with actuals but no valuation should be excluded."""
        # Only player 1 has a valuation
        valuations = [
            Valuation(
                player_id=1,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=40.0,
                rank=1,
                category_scores={"hr": 1.5, "r": 1.0},
            ),
        ]
        evaluator = _build_evaluator(valuations=valuations)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        # Only player 1 should be matched (players 2, 3, 4, 5 have actuals but no valuation)
        player_ids = {p.player_id for p in result.players}
        assert player_ids == {1}

    def test_surplus_is_predicted_minus_actual(self) -> None:
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        for p in result.players:
            assert p.surplus == pytest.approx(p.predicted_value - p.actual_value)

    def test_empty_season_returns_empty_result(self) -> None:
        evaluator = _build_evaluator(valuations=[], batting=[], pitching=[])
        result = evaluator.evaluate("zar", "1.0", 2099, _counting_league())
        assert result.n == 0
        assert result.value_mae == 0.0
        assert result.rank_correlation == 0.0
        assert result.players == []

    def test_player_names_resolved(self) -> None:
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        names = {p.player_name for p in result.players}
        assert "Mike Trout" in names
        assert "Aaron Judge" in names

    def test_zero_pa_batters_excluded_from_actuals(self) -> None:
        """Batters with pa=0 in actuals should be excluded from the valuation pool."""
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=35, r=100),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=500, hr=45, r=110),
            BattingStats(player_id=3, season=2025, source="fangraphs", pa=0, hr=0, r=0),
        ]
        evaluator = _build_evaluator(batting=batting)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        actual_player_ids = {p.player_id for p in result.players}
        assert 3 not in actual_player_ids

    def test_zero_ip_pitchers_excluded_from_actuals(self) -> None:
        """Pitchers with ip=0 in actuals should be excluded from the valuation pool."""
        pitching = [
            PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0),
            PitchingStats(player_id=5, season=2025, source="fangraphs", ip=0.0, w=0, sv=0),
        ]
        evaluator = _build_evaluator(pitching=pitching)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        actual_player_ids = {p.player_id for p in result.players}
        assert 5 not in actual_player_ids

    def test_two_way_player_both_roles_matched(self) -> None:
        """A two-way player (e.g. Ohtani) should appear in results as both batter and pitcher."""
        players = _standard_players() + [
            Player(id=6, name_first="Shohei", name_last="Ohtani"),
        ]
        valuations = _standard_valuations() + [
            Valuation(
                player_id=6,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=45.0,
                rank=1,
                category_scores={"hr": 2.0, "r": 1.5},
            ),
            Valuation(
                player_id=6,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="pitcher",
                position="p",
                value=20.0,
                rank=6,
                category_scores={"w": 0.5, "sv": 0.0},
            ),
        ]
        batting = _standard_batting_actuals() + [
            BattingStats(player_id=6, season=2025, source="fangraphs", pa=500, hr=40, r=90),
        ]
        pitching = _standard_pitching_actuals() + [
            PitchingStats(player_id=6, season=2025, source="fangraphs", ip=150.0, w=12, sv=0),
        ]
        appearances = _standard_appearances() + [
            PositionAppearance(player_id=6, season=2025, position="OF", games=120),
        ]
        evaluator = _build_evaluator(
            valuations=valuations,
            batting=batting,
            pitching=pitching,
            appearances=appearances,
            players=players,
        )
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())

        # Player 6 should appear twice: once as batter, once as pitcher
        ohtani_entries = [p for p in result.players if p.player_id == 6]
        assert len(ohtani_entries) == 2
        ohtani_types = {p.player_type for p in ohtani_entries}
        assert ohtani_types == {"batter", "pitcher"}

    def test_no_position_batter_with_roster_util_zero(self) -> None:
        """With roster_util=0, batters without position data still appear in results."""
        league = dataclasses.replace(_counting_league(), roster_util=0)
        # Only players 1 and 2 have position appearances; player 3 has none
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="OF", games=150),
            PositionAppearance(player_id=2, season=2025, position="OF", games=140),
        ]
        evaluator = _build_evaluator(appearances=appearances)
        result = evaluator.evaluate("zar", "1.0", 2025, league)
        actual_player_ids = {p.player_id for p in result.players}
        assert 3 in actual_player_ids

    def test_version_filter(self) -> None:
        """Two versions of valuations, filter returns correct one."""
        v1_valuations = _standard_valuations()
        v2_valuations = [
            Valuation(
                player_id=1,
                season=2025,
                system="zar",
                version="2.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=50.0,
                rank=1,
                category_scores={"hr": 2.0, "r": 1.5},
            ),
        ]
        all_vals = v1_valuations + v2_valuations
        evaluator = _build_evaluator(valuations=all_vals)
        result = evaluator.evaluate("zar", "2.0", 2025, _counting_league())
        # Only player 1 has a v2.0 valuation
        assert result.version == "2.0"
        player_ids = {p.player_id for p in result.players}
        assert player_ids == {1}
