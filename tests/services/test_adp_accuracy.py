import pytest

from fantasy_baseball_manager.domain.adp import ADP
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
from fantasy_baseball_manager.services.adp_accuracy import ADPAccuracyEvaluator
from tests.fakes.repos import (
    FakeADPRepo,
    FakeBattingStatsRepo,
    FakePlayerRepo,
    FakePitchingStatsRepo,
    FakePositionAppearanceRepo,
    FakeValuationRepo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _players() -> list[Player]:
    return [
        Player(id=1, name_first="Mike", name_last="Trout"),
        Player(id=2, name_first="Aaron", name_last="Judge"),
        Player(id=3, name_first="Mookie", name_last="Betts"),
        Player(id=4, name_first="Gerrit", name_last="Cole"),
        Player(id=5, name_first="Edwin", name_last="Diaz"),
    ]


def _batting_actuals(season: int = 2023) -> list[BattingStats]:
    return [
        BattingStats(player_id=1, season=season, source="fangraphs", pa=600, hr=45, r=110),
        BattingStats(player_id=2, season=season, source="fangraphs", pa=550, hr=35, r=100),
        BattingStats(player_id=3, season=season, source="fangraphs", pa=500, hr=20, r=80),
    ]


def _pitching_actuals(season: int = 2023) -> list[PitchingStats]:
    return [
        PitchingStats(player_id=4, season=season, source="fangraphs", ip=200.0, w=15, sv=0),
        PitchingStats(player_id=5, season=season, source="fangraphs", ip=70.0, w=5, sv=30),
    ]


def _appearances(season: int = 2023) -> list[PositionAppearance]:
    return [
        PositionAppearance(player_id=1, season=season, position="OF", games=150),
        PositionAppearance(player_id=2, season=season, position="OF", games=140),
        PositionAppearance(player_id=3, season=season, position="OF", games=130),
    ]


def _perfect_adp(season: int = 2023) -> list[ADP]:
    """ADP order matches actual value order: 1 > 2 > 3 > 4 > 5."""
    return [
        ADP(player_id=1, season=season, provider="fantasypros", overall_pick=1.0, rank=1, positions="OF"),
        ADP(player_id=2, season=season, provider="fantasypros", overall_pick=2.0, rank=2, positions="OF"),
        ADP(player_id=3, season=season, provider="fantasypros", overall_pick=3.0, rank=3, positions="OF"),
        ADP(player_id=4, season=season, provider="fantasypros", overall_pick=4.0, rank=4, positions="SP"),
        ADP(player_id=5, season=season, provider="fantasypros", overall_pick=5.0, rank=5, positions="RP"),
    ]


def _inverted_adp(season: int = 2023) -> list[ADP]:
    """ADP order is opposite to actual value order."""
    return [
        ADP(player_id=5, season=season, provider="fantasypros", overall_pick=1.0, rank=1, positions="RP"),
        ADP(player_id=4, season=season, provider="fantasypros", overall_pick=2.0, rank=2, positions="SP"),
        ADP(player_id=3, season=season, provider="fantasypros", overall_pick=3.0, rank=3, positions="OF"),
        ADP(player_id=2, season=season, provider="fantasypros", overall_pick=4.0, rank=4, positions="OF"),
        ADP(player_id=1, season=season, provider="fantasypros", overall_pick=5.0, rank=5, positions="OF"),
    ]


def _build_evaluator(
    adps: list[ADP] | None = None,
    batting: list[BattingStats] | None = None,
    pitching: list[PitchingStats] | None = None,
    appearances: list[PositionAppearance] | None = None,
    players: list[Player] | None = None,
    valuations: list[Valuation] | None = None,
) -> ADPAccuracyEvaluator:
    return ADPAccuracyEvaluator(
        adp_repo=FakeADPRepo(adps if adps is not None else _perfect_adp()),
        valuation_repo=FakeValuationRepo(valuations),
        player_repo=FakePlayerRepo(players if players is not None else _players()),
        batting_repo=FakeBattingStatsRepo(batting if batting is not None else _batting_actuals()),
        pitching_repo=FakePitchingStatsRepo(pitching if pitching is not None else _pitching_actuals()),
        position_repo=FakePositionAppearanceRepo(appearances if appearances is not None else _appearances()),
    )


# ---------------------------------------------------------------------------
# TestSingleSeason
# ---------------------------------------------------------------------------


class TestSingleSeason:
    def test_perfect_adp_correlation_near_one(self) -> None:
        evaluator = _build_evaluator()
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        assert result.rank_correlation > 0.8

    def test_inverted_adp_correlation_negative(self) -> None:
        evaluator = _build_evaluator(adps=_inverted_adp())
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        assert result.rank_correlation < 0.0

    def test_value_rmse_zero_when_perfect(self) -> None:
        """When ADP ordering identical to actual, implied == actual so RMSE = 0."""
        evaluator = _build_evaluator()
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        # With perfect ordering, value_error should be 0 for each player
        assert result.value_rmse == pytest.approx(0.0, abs=0.01)

    def test_value_rmse_positive_when_imperfect(self) -> None:
        evaluator = _build_evaluator(adps=_inverted_adp())
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        assert result.value_rmse > 0.0

    def test_top_n_precision_perfect_when_aligned(self) -> None:
        evaluator = _build_evaluator()
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        # With only 5 players, top-50/100/200 are all capped at 5 → precision should be 1.0
        for n in (50, 100, 200):
            assert result.top_n_precision[n] == pytest.approx(1.0)

    def test_n_matched_counts_intersection(self) -> None:
        evaluator = _build_evaluator()
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        assert result.n_matched == 5

    def test_players_without_actuals_excluded(self) -> None:
        """ADP player who didn't play excluded."""
        adps = _perfect_adp() + [
            ADP(player_id=99, season=2023, provider="fantasypros", overall_pick=6.0, rank=6, positions="OF"),
        ]
        evaluator = _build_evaluator(adps=adps)
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        player_ids = {p.player_id for p in result.players}
        assert 99 not in player_ids
        assert result.n_matched == 5

    def test_fewer_than_3_matches_returns_zero_correlation(self) -> None:
        """With < 3 matches, spearmanr can't compute → correlation = 0."""
        adps = [
            ADP(player_id=1, season=2023, provider="fantasypros", overall_pick=1.0, rank=1, positions="OF"),
            ADP(player_id=2, season=2023, provider="fantasypros", overall_pick=2.0, rank=2, positions="OF"),
        ]
        batting = [
            BattingStats(player_id=1, season=2023, source="fangraphs", pa=600, hr=45, r=110),
            BattingStats(player_id=2, season=2023, source="fangraphs", pa=550, hr=35, r=100),
        ]
        evaluator = _build_evaluator(adps=adps, batting=batting, pitching=[])
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        assert result.rank_correlation == 0.0
        assert result.n_matched == 2


# ---------------------------------------------------------------------------
# TestCompareSystem
# ---------------------------------------------------------------------------


class TestCompareSystem:
    def test_comparison_returns_system_metrics(self) -> None:
        valuations = [
            Valuation(
                player_id=1,
                season=2023,
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
            Valuation(
                player_id=2,
                season=2023,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=30.0,
                rank=2,
                category_scores={},
            ),
            Valuation(
                player_id=3,
                season=2023,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=20.0,
                rank=3,
                category_scores={},
            ),
            Valuation(
                player_id=4,
                season=2023,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="pitcher",
                position="p",
                value=25.0,
                rank=4,
                category_scores={},
            ),
            Valuation(
                player_id=5,
                season=2023,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="pitcher",
                position="p",
                value=15.0,
                rank=5,
                category_scores={},
            ),
        ]
        evaluator = _build_evaluator(valuations=valuations)
        report = evaluator.evaluate([2023], _counting_league(), compare_system=("zar", "1.0"))
        assert report.comparison is not None
        assert len(report.comparison) == 1
        sys_result = report.comparison[0]
        assert sys_result.system == "zar"
        assert sys_result.version == "1.0"
        assert sys_result.n_matched > 0

    def test_comparison_none_without_flag(self) -> None:
        evaluator = _build_evaluator()
        report = evaluator.evaluate([2023], _counting_league())
        assert report.comparison is None


# ---------------------------------------------------------------------------
# TestMultiSeason
# ---------------------------------------------------------------------------


class TestMultiSeason:
    def test_per_season_results_returned(self) -> None:
        adps = _perfect_adp(2022) + _perfect_adp(2023)
        batting = _batting_actuals(2022) + _batting_actuals(2023)
        pitching = _pitching_actuals(2022) + _pitching_actuals(2023)
        apps = _appearances(2022) + _appearances(2023)
        evaluator = _build_evaluator(adps=adps, batting=batting, pitching=pitching, appearances=apps)
        report = evaluator.evaluate([2022, 2023], _counting_league())
        assert len(report.adp_results) == 2
        assert report.adp_results[0].season == 2022
        assert report.adp_results[1].season == 2023

    def test_aggregate_is_mean_of_seasons(self) -> None:
        adps = _perfect_adp(2022) + _perfect_adp(2023)
        batting = _batting_actuals(2022) + _batting_actuals(2023)
        pitching = _pitching_actuals(2022) + _pitching_actuals(2023)
        apps = _appearances(2022) + _appearances(2023)
        evaluator = _build_evaluator(adps=adps, batting=batting, pitching=pitching, appearances=apps)
        report = evaluator.evaluate([2022, 2023], _counting_league())
        r0 = report.adp_results[0]
        r1 = report.adp_results[1]
        expected_mean_corr = (r0.rank_correlation + r1.rank_correlation) / 2
        assert report.mean_rank_correlation == pytest.approx(expected_mean_corr, abs=0.001)

    def test_season_with_no_adp_still_included(self) -> None:
        """Season with no ADP data returns result with n_matched=0."""
        evaluator = _build_evaluator(adps=[])
        report = evaluator.evaluate([2023], _counting_league())
        assert len(report.adp_results) == 1
        assert report.adp_results[0].n_matched == 0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_adp_returns_zero_metrics(self) -> None:
        evaluator = _build_evaluator(adps=[], batting=[], pitching=[])
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        assert result.n_matched == 0
        assert result.rank_correlation == 0.0
        assert result.value_rmse == 0.0
        assert result.value_mae == 0.0

    def test_top_n_capped_at_matched_count(self) -> None:
        """If only 3 batters matched, top-50 uses N=3."""
        adps = [
            ADP(player_id=1, season=2023, provider="fantasypros", overall_pick=1.0, rank=1, positions="OF"),
            ADP(player_id=2, season=2023, provider="fantasypros", overall_pick=2.0, rank=2, positions="OF"),
            ADP(player_id=3, season=2023, provider="fantasypros", overall_pick=3.0, rank=3, positions="OF"),
        ]
        evaluator = _build_evaluator(adps=adps, pitching=[])
        report = evaluator.evaluate([2023], _counting_league())
        result = report.adp_results[0]
        assert result.n_matched == 3
        # top-50 precision should be 1.0 since all 3 are both in ADP-top-3 and actual-top-3
        assert result.top_n_precision[50] == pytest.approx(1.0)
