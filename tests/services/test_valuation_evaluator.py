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
from fantasy_baseball_manager.domain.valuation import (
    Valuation,
    ValuationComparisonResult,
    ValuationEvalResult,
    check_valuation_regression,
)
from fantasy_baseball_manager.services.valuation_evaluator import ValuationEvaluator
from tests.fakes.repos import (
    FakeBattingStatsRepo,
    FakePitchingStatsRepo,
    FakePlayerRepo,
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

    def test_min_value_filter_reduces_n(self) -> None:
        """With min_value, only players with pred or actual > threshold are included."""
        evaluator = _build_evaluator()
        unfiltered = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        filtered = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), min_value=15.0)
        assert filtered.n < unfiltered.n
        # Every included player should have predicted > 15 or actual > 15
        for p in filtered.players:
            assert p.predicted_value > 15.0 or p.actual_value > 15.0

    def test_top_filter_reduces_n(self) -> None:
        """With top=3, only top 3 by predicted rank are included."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), top=3)
        assert result.n == 3
        # All included players should have predicted_rank <= 3
        for p in result.players:
            assert p.predicted_rank <= 3

    def test_both_filters_compose(self) -> None:
        """min_value applied first, then top limits the remaining set."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), min_value=15.0, top=2)
        # Should be at most 2
        assert result.n <= 2
        # And all should pass min_value
        for p in result.players:
            assert p.predicted_value > 15.0 or p.actual_value > 15.0

    def test_no_filter_backward_compatible(self) -> None:
        """Without filters, total_matched is None (backward compatible)."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.total_matched is None
        assert result.filter_description is None

    def test_min_value_all_filtered_returns_empty(self) -> None:
        """If min_value is so high nothing passes, result is empty."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), min_value=9999.0)
        assert result.n == 0
        assert result.players == []

    def test_total_matched_set_when_filtering(self) -> None:
        """total_matched reflects pre-filter count when filtering is active."""
        evaluator = _build_evaluator()
        unfiltered = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        filtered = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), min_value=15.0)
        assert filtered.total_matched is not None
        assert filtered.total_matched == unfiltered.n
        assert filtered.n < filtered.total_matched

    # -----------------------------------------------------------------------
    # WAR correlation tests
    # -----------------------------------------------------------------------

    def test_war_correlation_computed(self) -> None:
        """With WAR on all players, war_correlation is in [-1, 1]."""
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=35, r=100, war=5.0),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=45, r=110, war=7.0),
            BattingStats(player_id=3, season=2025, source="fangraphs", pa=500, hr=20, r=80, war=3.0),
        ]
        pitching = [
            PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0, war=4.0),
            PitchingStats(player_id=5, season=2025, source="fangraphs", ip=70.0, w=5, sv=30, war=2.0),
        ]
        evaluator = _build_evaluator(batting=batting, pitching=pitching)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.war_correlation is not None
        assert -1.0 <= result.war_correlation <= 1.0

    def test_war_correlation_batters_pitchers_separate(self) -> None:
        """Separate WAR ρ for batters and pitchers."""
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=35, r=100, war=5.0),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=45, r=110, war=7.0),
            BattingStats(player_id=3, season=2025, source="fangraphs", pa=500, hr=20, r=80, war=3.0),
        ]
        pitching = [
            PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0, war=4.0),
            PitchingStats(player_id=5, season=2025, source="fangraphs", ip=70.0, w=5, sv=30, war=2.0),
        ]
        evaluator = _build_evaluator(batting=batting, pitching=pitching)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.war_correlation_batters is not None
        assert -1.0 <= result.war_correlation_batters <= 1.0
        # Only 2 pitchers — need ≥3 for correlation
        assert result.war_correlation_pitchers is None

    def test_war_correlation_none_when_no_war_data(self) -> None:
        """All WAR=None → WAR fields stay None."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.war_correlation is None
        assert result.war_correlation_batters is None
        assert result.war_correlation_pitchers is None

    def test_war_correlation_none_when_target_excluded(self) -> None:
        """targets=frozenset({'hit-rate'}) → WAR fields None."""
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=35, r=100, war=5.0),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=45, r=110, war=7.0),
            BattingStats(player_id=3, season=2025, source="fangraphs", pa=500, hr=20, r=80, war=3.0),
        ]
        evaluator = _build_evaluator(batting=batting)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), targets=frozenset({"hit-rate"}))
        assert result.war_correlation is None
        assert result.war_correlation_batters is None
        assert result.war_correlation_pitchers is None

    def test_war_correlation_pitchers_too_few(self) -> None:
        """<3 pitchers with WAR → war_correlation_pitchers is None."""
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=35, r=100, war=5.0),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=45, r=110, war=7.0),
            BattingStats(player_id=3, season=2025, source="fangraphs", pa=500, hr=20, r=80, war=3.0),
        ]
        pitching = [
            PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0, war=4.0),
        ]
        evaluator = _build_evaluator(batting=batting, pitching=pitching)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.war_correlation_batters is not None
        assert result.war_correlation_pitchers is None

    # -----------------------------------------------------------------------
    # SP-only WAR ρ tests
    # -----------------------------------------------------------------------

    def test_war_correlation_sp_populated(self) -> None:
        """With ≥3 SP (gs≥5), war_correlation_sp is populated."""
        pitching = [
            PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0, war=4.0, gs=30),
            PitchingStats(player_id=5, season=2025, source="fangraphs", ip=180.0, w=12, sv=0, war=3.0, gs=28),
            PitchingStats(player_id=6, season=2025, source="fangraphs", ip=190.0, w=14, sv=0, war=3.5, gs=29),
        ]
        players = _standard_players() + [Player(id=6, name_first="Max", name_last="Scherzer")]
        valuations = _standard_valuations() + [
            Valuation(
                player_id=6,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="pitcher",
                position="p",
                value=18.0,
                rank=6,
                category_scores={"w": 0.8, "sv": 0.0},
            ),
        ]
        evaluator = _build_evaluator(valuations=valuations, pitching=pitching, players=players)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.war_correlation_sp is not None
        assert -1.0 <= result.war_correlation_sp <= 1.0

    def test_war_correlation_sp_none_when_few_sp(self) -> None:
        """With <3 SP, war_correlation_sp is None."""
        pitching = [
            PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0, war=4.0, gs=30),
            PitchingStats(player_id=5, season=2025, source="fangraphs", ip=70.0, w=5, sv=30, war=2.0, gs=0),
        ]
        evaluator = _build_evaluator(pitching=pitching)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        # Only 1 SP (gs≥5) — not enough
        assert result.war_correlation_sp is None

    def test_war_correlation_sp_excludes_relievers(self) -> None:
        """Relievers (gs<5) are excluded from SP WAR ρ."""
        pitching = [
            PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0, war=4.0, gs=30),
            PitchingStats(player_id=5, season=2025, source="fangraphs", ip=70.0, w=5, sv=30, war=2.0, gs=0),
            PitchingStats(player_id=6, season=2025, source="fangraphs", ip=60.0, w=3, sv=25, war=1.0, gs=2),
        ]
        players = _standard_players() + [Player(id=6, name_first="Edwin", name_last="Diaz2")]
        valuations = _standard_valuations() + [
            Valuation(
                player_id=6,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="pitcher",
                position="p",
                value=10.0,
                rank=6,
                category_scores={"w": -0.3, "sv": 1.0},
            ),
        ]
        evaluator = _build_evaluator(valuations=valuations, pitching=pitching, players=players)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        # Only 1 pitcher with gs>=5 → sp correlation is None
        assert result.war_correlation_sp is None

    # -----------------------------------------------------------------------
    # Top-N hit rate tests
    # -----------------------------------------------------------------------

    def test_hit_rate_computed(self) -> None:
        """With 5 players, hit_rates has keys ≤ 5 only."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.hit_rates is not None
        for n in result.hit_rates:
            assert n <= 5
            assert 0.0 <= result.hit_rates[n] <= 100.0

    def test_hit_rate_skipped_for_large_n(self) -> None:
        """With 5 players, N=25/50/100 are absent from hit_rates."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.hit_rates is not None
        assert 25 not in result.hit_rates
        assert 50 not in result.hit_rates
        assert 100 not in result.hit_rates

    def test_hit_rate_none_when_target_excluded(self) -> None:
        """targets=frozenset({'war'}) → hit_rates is None."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), targets=frozenset({"war"}))
        assert result.hit_rates is None

    def test_hit_rate_perfect_when_rankings_match(self) -> None:
        """Identical predicted/actual ranks → 100% hit rate."""
        # Use 3 batters with same ordering in predicted and actual
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
        ]
        # Actuals in same order: player 1 best, player 3 worst
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=50, r=120),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=30, r=90),
            BattingStats(player_id=3, season=2025, source="fangraphs", pa=500, hr=10, r=60),
        ]
        evaluator = _build_evaluator(valuations=valuations, batting=batting, pitching=[])
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), targets=frozenset({"hit-rate"}))
        assert result.hit_rates is not None
        # All hit rates should be 100% since ordering matches
        for _n, rate in result.hit_rates.items():
            assert rate == pytest.approx(100.0)

    # -----------------------------------------------------------------------
    # Per-category hit rate tests
    # -----------------------------------------------------------------------

    def test_category_hit_rates_computed(self) -> None:
        """With enough players, category_hit_rates is populated."""
        # Create 25 batters so top-20 hit rate can be computed for HR and R
        players: list[Player] = []
        valuations: list[Valuation] = []
        batting: list[BattingStats] = []
        appearances: list[PositionAppearance] = []
        for i in range(1, 26):
            players.append(Player(id=i, name_first=f"Player{i}", name_last="Test"))
            valuations.append(
                Valuation(
                    player_id=i,
                    season=2025,
                    system="zar",
                    version="1.0",
                    projection_system="steamer",
                    projection_version="v1",
                    player_type="batter",
                    position="of",
                    value=50.0 - i,
                    rank=i,
                    category_scores={"hr": 50.0 - i, "r": 100.0 - i},
                ),
            )
            batting.append(
                BattingStats(player_id=i, season=2025, source="fangraphs", pa=500, hr=50 - i, r=100 - i),
            )
            appearances.append(PositionAppearance(player_id=i, season=2025, position="OF", games=100))
        evaluator = _build_evaluator(
            valuations=valuations,
            batting=batting,
            pitching=[],
            appearances=appearances,
            players=players,
        )
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.category_hit_rates is not None
        assert "hr" in result.category_hit_rates
        assert "r" in result.category_hit_rates
        # Perfect ordering → 100% hit rate
        assert result.category_hit_rates["hr"] == pytest.approx(100.0)
        assert result.category_hit_rates["r"] == pytest.approx(100.0)

    def test_category_hit_rates_none_when_too_few_players(self) -> None:
        """With <20 players, category_hit_rates is None (not enough for top-20)."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        # Only 5 players — not enough for top-20 category hit rate
        assert result.category_hit_rates is None

    def test_category_hit_rates_empty_category_scores(self) -> None:
        """Players without category_scores are excluded from category hit rates."""
        # Create valuations without category_scores
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
                category_scores={},
            ),
        ]
        evaluator = _build_evaluator(valuations=valuations)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.category_hit_rates is None

    # -----------------------------------------------------------------------
    # Stratification tests
    # -----------------------------------------------------------------------

    def test_stratify_player_type_returns_cohorts(self) -> None:
        """stratify='player_type' produces cohorts with batter and pitcher keys."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), stratify="player_type")
        assert result.cohorts is not None
        assert "batter" in result.cohorts
        assert "pitcher" in result.cohorts

    def test_stratify_cohort_n_matches_player_count(self) -> None:
        """Each cohort n equals the number of that player type."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), stratify="player_type")
        assert result.cohorts is not None
        assert result.cohorts["batter"].n == 3  # 3 batters
        assert result.cohorts["pitcher"].n == 2  # 2 pitchers

    def test_stratify_cohort_metrics_independent(self) -> None:
        """Each cohort has its own MAE and ρ (not the overall values)."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), stratify="player_type")
        assert result.cohorts is not None
        batter_cohort = result.cohorts["batter"]
        pitcher_cohort = result.cohorts["pitcher"]
        # Cohort MAE should differ from overall (different populations)
        assert batter_cohort.value_mae != result.value_mae or pitcher_cohort.value_mae != result.value_mae

    def test_stratify_none_no_cohorts(self) -> None:
        """Without stratify, cohorts is None (backward compat)."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.cohorts is None

    def test_stratify_composes_with_min_value(self) -> None:
        """Cohorts respect the min_value population filter."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), min_value=15.0, stratify="player_type")
        assert result.cohorts is not None
        # Total cohort n should equal filtered n
        cohort_total = sum(c.n for c in result.cohorts.values())
        assert cohort_total == result.n

    def test_stratify_includes_war_correlation(self) -> None:
        """Cohort results include WAR ρ when WAR data is present."""
        batting = [
            BattingStats(player_id=1, season=2025, source="fangraphs", pa=600, hr=35, r=100, war=5.0),
            BattingStats(player_id=2, season=2025, source="fangraphs", pa=550, hr=45, r=110, war=7.0),
            BattingStats(player_id=3, season=2025, source="fangraphs", pa=500, hr=20, r=80, war=3.0),
        ]
        pitching = [
            PitchingStats(player_id=4, season=2025, source="fangraphs", ip=200.0, w=15, sv=0, war=4.0),
            PitchingStats(player_id=5, season=2025, source="fangraphs", ip=70.0, w=5, sv=30, war=2.0),
        ]
        evaluator = _build_evaluator(batting=batting, pitching=pitching)
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), stratify="player_type")
        assert result.cohorts is not None
        # Batter cohort has 3 players with WAR — should have war_correlation
        assert result.cohorts["batter"].war_correlation is not None

    # -----------------------------------------------------------------------
    # Tail accuracy tests
    # -----------------------------------------------------------------------

    def test_tail_results_computed(self) -> None:
        """With 5 players and tail_ns=(3, 5), result has tail_results with keys 3 and 5."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), tail_ns=(3, 5))
        assert result.tail_results is not None
        assert 3 in result.tail_results
        assert 5 in result.tail_results

    def test_tail_results_skip_large_n(self) -> None:
        """With 5 players, tail_ns=(3, 50) → only key 3 present."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), tail_ns=(3, 50))
        assert result.tail_results is not None
        assert 3 in result.tail_results
        assert 50 not in result.tail_results

    def test_tail_none_by_default(self) -> None:
        """Without tail_ns, tail_results is None."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league())
        assert result.tail_results is None

    def test_tail_composes_with_stratify(self) -> None:
        """tail + stratify both active, both populated."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), stratify="player_type", tail_ns=(3, 5))
        assert result.cohorts is not None
        assert result.tail_results is not None
        assert 3 in result.tail_results

    def test_tail_composes_with_min_value(self) -> None:
        """Tail respects population filter."""
        evaluator = _build_evaluator()
        result = evaluator.evaluate("zar", "1.0", 2025, _counting_league(), min_value=15.0, tail_ns=(3,))
        assert result.tail_results is not None
        if 3 in result.tail_results:
            assert result.tail_results[3].n == 3

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


# ---------------------------------------------------------------------------
# Compare tests
# ---------------------------------------------------------------------------


class TestValuationCompare:
    def test_compare_returns_both_results(self) -> None:
        evaluator = _build_evaluator()
        result = evaluator.compare(
            "zar",
            "1.0",
            "zar",
            "1.0",
            2025,
            _counting_league(),
            min_value=None,
            top=None,
            targets=None,
            stratify=None,
            tail_ns=None,
        )
        assert isinstance(result, ValuationComparisonResult)
        assert result.season == 2025
        assert result.baseline.system == "zar"
        assert result.candidate.system == "zar"
        assert result.baseline.n == 5
        assert result.candidate.n == 5

    def test_compare_passes_filters(self) -> None:
        evaluator = _build_evaluator()
        result = evaluator.compare(
            "zar",
            "1.0",
            "zar",
            "1.0",
            2025,
            _counting_league(),
            min_value=15.0,
            top=None,
            targets=frozenset({"war"}),
            stratify="player_type",
            tail_ns=(3,),
        )
        # Both results should have the same filtered n
        assert result.baseline.n == result.candidate.n
        # min_value filter should reduce population
        assert result.baseline.n < 5
        # stratify should produce cohorts
        assert result.baseline.cohorts is not None
        # targets=war should exclude hit-rate
        assert result.baseline.hit_rates is None


# ---------------------------------------------------------------------------
# Regression check tests
# ---------------------------------------------------------------------------


def _make_eval_result(
    *,
    war_correlation: float | None = None,
    war_correlation_batters: float | None = None,
    war_correlation_pitchers: float | None = None,
    hit_rates: dict[int, float] | None = None,
) -> ValuationEvalResult:
    return ValuationEvalResult(
        system="test",
        version="1.0",
        season=2025,
        value_mae=10.0,
        rank_correlation=0.5,
        n=100,
        players=[],
        war_correlation=war_correlation,
        war_correlation_batters=war_correlation_batters,
        war_correlation_pitchers=war_correlation_pitchers,
        hit_rates=hit_rates,
    )


class TestValuationRegressionCheck:
    def test_passes_when_no_drop(self) -> None:
        baseline = _make_eval_result(war_correlation=0.20, hit_rates={25: 30.0, 50: 40.0})
        candidate = _make_eval_result(war_correlation=0.22, hit_rates={25: 32.0, 50: 42.0})
        result = check_valuation_regression(baseline, candidate)
        assert result.passed is True
        assert result.war_passed is True
        assert result.hit_rate_passed is True

    def test_fails_on_war_drop(self) -> None:
        baseline = _make_eval_result(war_correlation=0.20, hit_rates={25: 30.0, 50: 40.0})
        candidate = _make_eval_result(war_correlation=0.15, hit_rates={25: 30.0, 50: 40.0})
        result = check_valuation_regression(baseline, candidate)
        assert result.passed is False
        assert result.war_passed is False
        assert result.hit_rate_passed is True

    def test_fails_on_hit_rate_drop(self) -> None:
        baseline = _make_eval_result(war_correlation=0.20, hit_rates={25: 40.0, 50: 50.0})
        candidate = _make_eval_result(war_correlation=0.20, hit_rates={25: 30.0, 50: 40.0})
        result = check_valuation_regression(baseline, candidate)
        assert result.passed is False
        assert result.war_passed is True
        assert result.hit_rate_passed is False

    def test_handles_none_war(self) -> None:
        baseline = _make_eval_result(war_correlation=None, hit_rates={25: 30.0})
        candidate = _make_eval_result(war_correlation=None, hit_rates={25: 30.0})
        result = check_valuation_regression(baseline, candidate)
        assert result.passed is True
        assert result.war_passed is True

    def test_handles_none_hit_rates(self) -> None:
        baseline = _make_eval_result(war_correlation=0.20, hit_rates=None)
        candidate = _make_eval_result(war_correlation=0.20, hit_rates=None)
        result = check_valuation_regression(baseline, candidate)
        assert result.passed is True
        assert result.hit_rate_passed is True

    def test_war_drop_at_threshold_passes(self) -> None:
        """Exactly 0.01 drop is allowed (<=)."""
        baseline = _make_eval_result(war_correlation=0.20, hit_rates={25: 30.0})
        candidate = _make_eval_result(war_correlation=0.19, hit_rates={25: 30.0})
        result = check_valuation_regression(baseline, candidate)
        assert result.war_passed is True

    def test_hit_rate_drop_at_threshold_passes(self) -> None:
        """Exactly 5pp drop is allowed (<=)."""
        baseline = _make_eval_result(war_correlation=0.20, hit_rates={25: 35.0})
        candidate = _make_eval_result(war_correlation=0.20, hit_rates={25: 30.0})
        result = check_valuation_regression(baseline, candidate)
        assert result.hit_rate_passed is True
