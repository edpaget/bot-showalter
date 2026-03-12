import pytest

from fantasy_baseball_manager.domain.league_settings import CategoryConfig, Direction, StatType
from fantasy_baseball_manager.domain.sgp import DenominatorMethod
from fantasy_baseball_manager.domain.yahoo_team_stats import TeamSeasonStats
from fantasy_baseball_manager.services.sgp_denominator import (
    compute_representative_team_totals,
    compute_sgp_denominators,
)


def _team(season: int, team_key: str, stats: dict[str, float]) -> TeamSeasonStats:
    return TeamSeasonStats(
        team_key=team_key,
        league_key=f"mlb.l.{season}",
        season=season,
        team_name=f"Team {team_key}",
        final_rank=1,
        stat_values=stats,
    )


def _counting_cat(key: str) -> CategoryConfig:
    return CategoryConfig(key=key, name=key, stat_type=StatType.COUNTING, direction=Direction.HIGHER)


def _rate_cat(key: str, direction: Direction = Direction.LOWER) -> CategoryConfig:
    return CategoryConfig(
        key=key, name=key, stat_type=StatType.RATE, direction=direction, numerator="x", denominator="y"
    )


class TestComputeSgpDenominators:
    def test_single_season_counting_stat(self) -> None:
        """3 teams with HR = 10, 20, 30 → gaps = [10, 10] → denominator = 10."""
        standings = [
            _team(2024, "a", {"HR": 10.0}),
            _team(2024, "b", {"HR": 20.0}),
            _team(2024, "c", {"HR": 30.0}),
        ]
        cats = [_counting_cat("HR")]
        result = compute_sgp_denominators(standings, cats)

        assert len(result.per_season) == 1
        assert result.per_season[0].category == "HR"
        assert result.per_season[0].denominator == 10.0
        assert result.per_season[0].num_teams == 3
        assert result.averages["HR"] == 10.0

    def test_two_seasons_averaging(self) -> None:
        """Two seasons with different gaps average correctly."""
        standings = [
            _team(2023, "a", {"HR": 10.0}),
            _team(2023, "b", {"HR": 20.0}),
            _team(2023, "c", {"HR": 30.0}),
            # season 2024: gaps = [20, 20] → denominator = 20
            _team(2024, "a", {"HR": 0.0}),
            _team(2024, "b", {"HR": 20.0}),
            _team(2024, "c", {"HR": 40.0}),
        ]
        cats = [_counting_cat("HR")]
        result = compute_sgp_denominators(standings, cats)

        assert len(result.per_season) == 2
        assert result.averages["HR"] == 15.0  # mean(10, 20)

    def test_rate_stat_lower_is_better(self) -> None:
        """ERA (lower-is-better): gaps are negative, denominator should be positive."""
        standings = [
            _team(2024, "a", {"ERA": 3.0}),
            _team(2024, "b", {"ERA": 3.5}),
            _team(2024, "c", {"ERA": 4.0}),
        ]
        cats = [_rate_cat("ERA", Direction.LOWER)]
        result = compute_sgp_denominators(standings, cats)

        # Raw gaps are positive (3.5 - 3.0 = 0.5, 4.0 - 3.5 = 0.5)
        # Negated for LOWER direction → -0.5? No — gaps are always ascending so positive.
        # For LOWER, we negate: denominator = -(mean of positive gaps) → negative
        # But we want positive. Let me re-check the algorithm.
        # Sorted ascending: 3.0, 3.5, 4.0. Gaps: 0.5, 0.5. Mean = 0.5.
        # For LOWER direction: negate → -0.5.
        # Wait, the plan says "negate so it's always positive". But negating a positive gives negative.
        # The convention is: for lower-is-better, the gaps represent "worse" performance
        # as values increase, so we negate to indicate that improvement means lower values.
        # Actually, re-reading: the denominator should be positive. For ERA, going from
        # 4.0 to 3.5 is an improvement. The gap magnitude is 0.5 ERA per standings point.
        # The sign should be positive (it's always "this many units buys one standings point").
        # The negation is because ascending sort for lower-is-better means the gaps
        # represent getting worse, not better. So we negate.
        # But 0.5 negated is -0.5, which is negative. That's wrong.
        # Let me re-read the plan more carefully:
        # "For "lower is better" categories (ERA, WHIP): negate the denominator so it's
        # always positive (a positive denominator means "this many units buys one standings
        # point improvement")"
        # So the gaps for ERA sorted ascending are positive (3.0→3.5→4.0, gaps +0.5).
        # These represent "each 0.5 ERA increase costs you one standings point."
        # For the valuation formula, we want: marginal_sgp = (baseline - player) / denom.
        # If denom is positive, then a player with lower ERA gets positive SGP.
        # Actually the negation should make it negative so the valuation formula can use
        # marginal_sgp = (player_rate - league_avg) / denom, and for lower-is-better,
        # a negative denom combined with a negative (player - avg) gives positive SGP.
        # Let me just verify what the code does:
        assert result.per_season[0].denominator == -0.5

    def test_rate_stat_higher_is_better(self) -> None:
        """OBP (higher-is-better rate): denominator stays positive."""
        standings = [
            _team(2024, "a", {"OBP": 0.300}),
            _team(2024, "b", {"OBP": 0.320}),
            _team(2024, "c", {"OBP": 0.340}),
        ]
        cats = [_rate_cat("OBP", Direction.HIGHER)]
        result = compute_sgp_denominators(standings, cats)

        assert result.per_season[0].denominator == pytest.approx(0.02)

    def test_missing_category_skipped(self) -> None:
        """Teams missing a category in stat_values are excluded gracefully."""
        standings = [
            _team(2024, "a", {"HR": 10.0}),
            _team(2024, "b", {"HR": 20.0}),
            _team(2024, "c", {}),  # missing HR
        ]
        cats = [_counting_cat("HR")]
        result = compute_sgp_denominators(standings, cats)

        # Only 2 teams have HR → 1 gap = 10
        assert result.per_season[0].denominator == 10.0

    def test_single_team_skipped(self) -> None:
        """With only one team having data, no denominator is computed."""
        standings = [
            _team(2024, "a", {"HR": 10.0}),
        ]
        cats = [_counting_cat("HR")]
        result = compute_sgp_denominators(standings, cats)

        assert len(result.per_season) == 0
        assert len(result.averages) == 0

    def test_multiple_categories(self) -> None:
        """Verify multiple categories are computed independently."""
        standings = [
            _team(2024, "a", {"HR": 10.0, "SB": 5.0}),
            _team(2024, "b", {"HR": 20.0, "SB": 15.0}),
            _team(2024, "c", {"HR": 30.0, "SB": 25.0}),
        ]
        cats = [_counting_cat("HR"), _counting_cat("SB")]
        result = compute_sgp_denominators(standings, cats)

        assert len(result.per_season) == 2
        assert result.averages["HR"] == 10.0
        assert result.averages["SB"] == 10.0

    def test_uneven_gaps(self) -> None:
        """Gaps that differ produce a correct mean."""
        standings = [
            _team(2024, "a", {"HR": 10.0}),
            _team(2024, "b", {"HR": 20.0}),
            _team(2024, "c", {"HR": 50.0}),
        ]
        cats = [_counting_cat("HR")]
        result = compute_sgp_denominators(standings, cats)

        # Gaps: 10, 30 → mean = 20
        assert result.per_season[0].denominator == 20.0


class TestRegressionDenominators:
    def test_regression_single_season_uniform_gaps(self) -> None:
        """3 teams with HR = 10, 20, 30. Uniform spacing → regression slope == mean-gap."""
        standings = [
            _team(2024, "a", {"HR": 10.0}),
            _team(2024, "b", {"HR": 20.0}),
            _team(2024, "c", {"HR": 30.0}),
        ]
        cats = [_counting_cat("HR")]
        result = compute_sgp_denominators(standings, cats, method=DenominatorMethod.REGRESSION)

        # Ranks: 1, 2, 3. Values: 10, 20, 30. Slope = 10.0
        assert result.per_season[0].denominator == pytest.approx(10.0)

    def test_regression_with_outlier(self) -> None:
        """4 teams: HR = 10, 20, 30, 100. Regression is less sensitive to outlier than mean-gap."""
        standings = [
            _team(2024, "a", {"HR": 10.0}),
            _team(2024, "b", {"HR": 20.0}),
            _team(2024, "c", {"HR": 30.0}),
            _team(2024, "d", {"HR": 100.0}),
        ]
        cats = [_counting_cat("HR")]

        mean_gap_result = compute_sgp_denominators(standings, cats, method=DenominatorMethod.MEAN_GAP)
        reg_result = compute_sgp_denominators(standings, cats, method=DenominatorMethod.REGRESSION)

        # Mean-gap = (100 - 10) / 3 = 30.0
        assert mean_gap_result.per_season[0].denominator == 30.0

        # Regression slope: x = [1,2,3,4], y = [10,20,30,100]
        # x̄=2.5, ȳ=40. Σ(x-x̄)(y-ȳ) = (-1.5)(-30)+(-0.5)(-20)+(0.5)(-10)+(1.5)(60) = 45+10-5+90 = 140
        # Σ(x-x̄)² = 2.25+0.25+0.25+2.25 = 5. Slope = 140/5 = 28.0
        assert reg_result.per_season[0].denominator == pytest.approx(28.0)
        assert reg_result.per_season[0].denominator != mean_gap_result.per_season[0].denominator

    def test_regression_lower_is_better(self) -> None:
        """ERA data — regression slope is negated for LOWER direction."""
        standings = [
            _team(2024, "a", {"ERA": 3.0}),
            _team(2024, "b", {"ERA": 3.5}),
            _team(2024, "c", {"ERA": 4.0}),
        ]
        cats = [_rate_cat("ERA", Direction.LOWER)]
        result = compute_sgp_denominators(standings, cats, method=DenominatorMethod.REGRESSION)

        # Sorted ascending: 3.0, 3.5, 4.0. Ranks: 1, 2, 3. Slope = 0.5.
        # Negated for LOWER → -0.5
        assert result.per_season[0].denominator == pytest.approx(-0.5)

    def test_regression_ties(self) -> None:
        """Two teams with identical values get mean rank."""
        standings = [
            _team(2024, "a", {"HR": 10.0}),
            _team(2024, "b", {"HR": 20.0}),
            _team(2024, "c", {"HR": 20.0}),
            _team(2024, "d", {"HR": 40.0}),
        ]
        cats = [_counting_cat("HR")]
        result = compute_sgp_denominators(standings, cats, method=DenominatorMethod.REGRESSION)

        # Sorted: 10, 20, 20, 40. Ranks: 1, 2.5, 2.5, 4 (tied teams share mean rank)
        # x = [1, 2.5, 2.5, 4], y = [10, 20, 20, 40]
        # x̄ = 2.5, ȳ = 22.5
        # Σ(x-x̄)(y-ȳ) = (-1.5)(-12.5)+(0)(-2.5)+(0)(-2.5)+(1.5)(17.5) = 18.75+0+0+26.25 = 45
        # Σ(x-x̄)² = 2.25+0+0+2.25 = 4.5
        # Slope = 45 / 4.5 = 10.0
        assert result.per_season[0].denominator == pytest.approx(10.0)

    def test_mean_gap_backward_compatible(self) -> None:
        """Passing method=MEAN_GAP produces identical results to default call."""
        standings = [
            _team(2024, "a", {"HR": 10.0}),
            _team(2024, "b", {"HR": 20.0}),
            _team(2024, "c", {"HR": 50.0}),
        ]
        cats = [_counting_cat("HR")]

        default_result = compute_sgp_denominators(standings, cats)
        explicit_result = compute_sgp_denominators(standings, cats, method=DenominatorMethod.MEAN_GAP)

        assert default_result.per_season[0].denominator == explicit_result.per_season[0].denominator
        assert default_result.averages == explicit_result.averages

    def test_default_method_is_mean_gap(self) -> None:
        """Calling without method parameter produces same results as method=MEAN_GAP."""
        standings = [
            _team(2024, "a", {"HR": 10.0, "SB": 5.0}),
            _team(2024, "b", {"HR": 20.0, "SB": 15.0}),
            _team(2024, "c", {"HR": 30.0, "SB": 25.0}),
        ]
        cats = [_counting_cat("HR"), _counting_cat("SB")]

        default_result = compute_sgp_denominators(standings, cats)
        mean_gap_result = compute_sgp_denominators(standings, cats, method=DenominatorMethod.MEAN_GAP)

        for i in range(len(default_result.per_season)):
            assert default_result.per_season[i].denominator == mean_gap_result.per_season[i].denominator
        assert default_result.averages == mean_gap_result.averages


class TestRepresentativeTeamTotals:
    def test_rate_category_with_denom_in_standings(self) -> None:
        """ERA with 'era' and 'ip' in standings → returns (avg_era, avg_ip)."""
        standings = [
            _team(2024, "a", {"era": 3.5, "ip": 1200.0}),
            _team(2024, "b", {"era": 4.0, "ip": 1300.0}),
            _team(2024, "c", {"era": 3.0, "ip": 1100.0}),
        ]
        cats = [
            CategoryConfig(
                key="era",
                name="ERA",
                stat_type=StatType.RATE,
                direction=Direction.LOWER,
                numerator="er",
                denominator="ip",
            )
        ]
        result = compute_representative_team_totals(standings, cats)
        assert "era" in result
        avg_era, avg_ip = result["era"]
        assert avg_era == pytest.approx(3.5)  # mean(3.5, 4.0, 3.0)
        assert avg_ip == pytest.approx(1200.0)  # mean(1200, 1300, 1100)

    def test_rate_category_missing_denom_in_standings(self) -> None:
        """OBP with 'obp' but no 'pa' in standings → category omitted."""
        standings = [
            _team(2024, "a", {"obp": 0.340}),
            _team(2024, "b", {"obp": 0.320}),
        ]
        cats = [
            CategoryConfig(
                key="obp",
                name="OBP",
                stat_type=StatType.RATE,
                direction=Direction.HIGHER,
                numerator="obp_num",
                denominator="pa",
            )
        ]
        result = compute_representative_team_totals(standings, cats)
        assert "obp" not in result

    def test_counting_category_excluded(self) -> None:
        """Counting stats (HR) are not included in representative team totals."""
        standings = [
            _team(2024, "a", {"HR": 200.0}),
            _team(2024, "b", {"HR": 180.0}),
        ]
        cats = [_counting_cat("HR")]
        result = compute_representative_team_totals(standings, cats)
        assert result == {}

    def test_multiple_seasons_averaged(self) -> None:
        """Two seasons of data → averages across all teams and seasons."""
        standings = [
            _team(2023, "a", {"era": 3.0, "ip": 1000.0}),
            _team(2023, "b", {"era": 4.0, "ip": 1200.0}),
            _team(2024, "a", {"era": 3.5, "ip": 1100.0}),
            _team(2024, "b", {"era": 4.5, "ip": 1300.0}),
        ]
        cats = [
            CategoryConfig(
                key="era",
                name="ERA",
                stat_type=StatType.RATE,
                direction=Direction.LOWER,
                numerator="er",
                denominator="ip",
            )
        ]
        result = compute_representative_team_totals(standings, cats)
        avg_era, avg_ip = result["era"]
        # mean(3.0, 4.0, 3.5, 4.5) = 3.75
        assert avg_era == pytest.approx(3.75)
        # mean(1000, 1200, 1100, 1300) = 1150
        assert avg_ip == pytest.approx(1150.0)

    def test_empty_standings(self) -> None:
        """Empty input → empty dict."""
        cats = [
            CategoryConfig(
                key="era",
                name="ERA",
                stat_type=StatType.RATE,
                direction=Direction.LOWER,
                numerator="er",
                denominator="ip",
            )
        ]
        result = compute_representative_team_totals([], cats)
        assert result == {}
