import pytest

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.services.category_tracker import analyze_roster


def _league(
    *,
    teams: int = 12,
    batting: tuple[CategoryConfig, ...] = (),
    pitching: tuple[CategoryConfig, ...] = (),
) -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=teams,
        budget=260,
        roster_batters=14,
        roster_pitchers=10,
        batting_categories=batting,
        pitching_categories=pitching,
    )


def _proj(
    player_id: int,
    player_type: str,
    stats: dict[str, float],
) -> Projection:
    return Projection(
        player_id=player_id,
        season=2026,
        system="steamer",
        version="1.0",
        player_type=player_type,
        stat_json=stats,
    )


HR_CAT = CategoryConfig(key="hr", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
RBI_CAT = CategoryConfig(key="rbi", name="RBI", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
SB_CAT = CategoryConfig(key="sb", name="Stolen Bases", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
AVG_CAT = CategoryConfig(
    key="avg",
    name="Batting Average",
    stat_type=StatType.RATE,
    direction=Direction.HIGHER,
    numerator="h",
    denominator="ab",
)
OBP_CAT = CategoryConfig(
    key="obp",
    name="On-Base Percentage",
    stat_type=StatType.RATE,
    direction=Direction.HIGHER,
    numerator="bb+h",
    denominator="pa",
)
ERA_CAT = CategoryConfig(
    key="era",
    name="ERA",
    stat_type=StatType.RATE,
    direction=Direction.LOWER,
    numerator="er",
    denominator="ip",
    # ERA = (ER / IP) * 9, but we test the raw ratio here; the actual
    # display scaling is out of scope for phase 1.
)
K_CAT = CategoryConfig(key="k", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
WHIP_CAT = CategoryConfig(
    key="whip",
    name="WHIP",
    stat_type=StatType.RATE,
    direction=Direction.LOWER,
    numerator="bb+h",
    denominator="ip",
)


class TestCountingStats:
    def test_sums_correctly(self) -> None:
        """Two batters' HR totals are summed."""
        league = _league(batting=(HR_CAT,))
        projections = [
            _proj(1, "batter", {"hr": 35}),
            _proj(2, "batter", {"hr": 25}),
        ]
        # All projections in the pool, both on the roster
        result = analyze_roster([1, 2], projections, league)
        hr_proj = result.projections[0]
        assert hr_proj.category == "hr"
        assert hr_proj.projected_value == pytest.approx(60.0)

    def test_missing_stat_treated_as_zero(self) -> None:
        """A player without the stat in stat_json contributes 0."""
        league = _league(batting=(HR_CAT,))
        projections = [
            _proj(1, "batter", {"hr": 30}),
            _proj(2, "batter", {"rbi": 100}),  # no 'hr' key
        ]
        result = analyze_roster([1, 2], projections, league)
        assert result.projections[0].projected_value == pytest.approx(30.0)


class TestRateStats:
    def test_weighted_average(self) -> None:
        """AVG = total H / total AB, not average of individual AVGs."""
        league = _league(batting=(AVG_CAT,))
        # Player 1: 160 H / 500 AB = .320
        # Player 2: 120 H / 500 AB = .240
        # Correct weighted: 280 / 1000 = .280 (not (.320+.240)/2 = .280 — same here,
        # but differs with unequal AB)
        projections = [
            _proj(1, "batter", {"h": 160, "ab": 500}),
            _proj(2, "batter", {"h": 120, "ab": 500}),
        ]
        result = analyze_roster([1, 2], projections, league)
        assert result.projections[0].projected_value == pytest.approx(0.280)

    def test_weighted_average_unequal_denominator(self) -> None:
        """With unequal AB, simple average would be wrong."""
        league = _league(batting=(AVG_CAT,))
        # Player 1: 180 H / 600 AB = .300
        # Player 2: 40 H / 200 AB = .200
        # Weighted: 220 / 800 = .275 (simple avg would be .250)
        projections = [
            _proj(1, "batter", {"h": 180, "ab": 600}),
            _proj(2, "batter", {"h": 40, "ab": 200}),
        ]
        result = analyze_roster([1, 2], projections, league)
        assert result.projections[0].projected_value == pytest.approx(0.275)

    def test_compound_numerator(self) -> None:
        """OBP uses numerator 'bb+h' — compound expression."""
        league = _league(batting=(OBP_CAT,))
        projections = [
            _proj(1, "batter", {"bb": 60, "h": 150, "pa": 600}),
            _proj(2, "batter", {"bb": 40, "h": 130, "pa": 500}),
        ]
        # Numerator: (60+150) + (40+130) = 380
        # Denominator: 600 + 500 = 1100
        # OBP: 380 / 1100 ≈ 0.34545
        result = analyze_roster([1, 2], projections, league)
        assert result.projections[0].projected_value == pytest.approx(380 / 1100)

    def test_zero_denominator_returns_zero(self) -> None:
        """If no AB at all, rate should be 0.0 not an error."""
        league = _league(batting=(AVG_CAT,))
        projections = [
            _proj(1, "batter", {"h": 0, "ab": 0}),
        ]
        result = analyze_roster([1], projections, league)
        assert result.projections[0].projected_value == pytest.approx(0.0)


class TestLeagueRankEstimation:
    def test_above_average_team_ranks_top_half(self) -> None:
        """A team with more HR than league average should rank in top half."""
        league = _league(teams=12, batting=(HR_CAT,))
        # 12 teams worth of projections, each with 20 HR
        projections = [_proj(i, "batter", {"hr": 20.0}) for i in range(1, 169)]
        # Our roster of 14 batters: each has 30 HR (well above avg of 20)
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"hr": 30.0}) for pid in roster_ids]
        result = analyze_roster(roster_ids, projections, league)
        # Should rank in top half (1-6 of 12)
        assert result.projections[0].league_rank_estimate <= 6

    def test_below_average_team_ranks_bottom_half(self) -> None:
        """A team with fewer HR than average should rank in bottom half."""
        league = _league(teams=12, batting=(HR_CAT,))
        projections = [_proj(i, "batter", {"hr": 20.0}) for i in range(1, 169)]
        # Our roster: each has 10 HR (below avg of 20)
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"hr": 10.0}) for pid in roster_ids]
        result = analyze_roster(roster_ids, projections, league)
        assert result.projections[0].league_rank_estimate >= 7

    def test_rank_clamped_to_valid_range(self) -> None:
        """Rank should be between 1 and N teams inclusive."""
        league = _league(teams=12, batting=(HR_CAT,))
        projections = [_proj(i, "batter", {"hr": 20.0}) for i in range(1, 169)]
        # Extremely high HR team
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"hr": 100.0}) for pid in roster_ids]
        result = analyze_roster(roster_ids, projections, league)
        rank = result.projections[0].league_rank_estimate
        assert 1 <= rank <= 12


class TestStrengthClassification:
    def test_strong_category(self) -> None:
        """Top third rank → 'strong'."""
        league = _league(teams=12, batting=(HR_CAT,))
        projections = [_proj(i, "batter", {"hr": 20.0}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"hr": 30.0}) for pid in roster_ids]
        result = analyze_roster(roster_ids, projections, league)
        assert result.projections[0].strength == "strong"

    def test_weak_category(self) -> None:
        """Bottom third rank → 'weak'."""
        league = _league(teams=12, batting=(HR_CAT,))
        projections = [_proj(i, "batter", {"hr": 20.0}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"hr": 10.0}) for pid in roster_ids]
        result = analyze_roster(roster_ids, projections, league)
        assert result.projections[0].strength == "weak"

    def test_average_category(self) -> None:
        """Middle third rank → 'average'."""
        league = _league(teams=12, batting=(HR_CAT,))
        projections = [_proj(i, "batter", {"hr": 20.0}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Slightly below avg → should be middle range
        projections += [_proj(pid, "batter", {"hr": 19.0}) for pid in roster_ids]
        result = analyze_roster(roster_ids, projections, league)
        assert result.projections[0].strength == "average"


class TestPitchingCategories:
    def test_pitching_counting_stat(self) -> None:
        """Pitching strikeouts (counting, HIGHER) sum correctly."""
        league = _league(pitching=(K_CAT,))
        projections = [
            _proj(1, "pitcher", {"k": 200}),
            _proj(2, "pitcher", {"k": 150}),
        ]
        result = analyze_roster([1, 2], projections, league)
        assert result.projections[0].projected_value == pytest.approx(350.0)

    def test_era_lower_is_better(self) -> None:
        """ERA (LOWER direction): below-avg ERA → better rank (lower number)."""
        # League pool: 120 pitchers with ER=40, IP=200 (ratio 0.2)
        projections = [_proj(i, "pitcher", {"er": 40, "ip": 200}) for i in range(1, 121)]
        # Our roster: 10 pitchers with ER=20, IP=200 (ratio 0.1 — much better)
        roster_ids = list(range(200, 210))
        projections += [_proj(pid, "pitcher", {"er": 20, "ip": 200}) for pid in roster_ids]
        league = _league(teams=12, pitching=(ERA_CAT,))
        result = analyze_roster(roster_ids, projections, league)
        # Better ERA should rank in top half
        assert result.projections[0].league_rank_estimate <= 6
        assert result.projections[0].strength == "strong"


class TestMixedCategories:
    def test_batting_and_pitching_combined(self) -> None:
        """Analysis includes both batting and pitching categories."""
        league = _league(
            teams=12,
            batting=(HR_CAT,),
            pitching=(K_CAT,),
        )
        projections = [
            _proj(1, "batter", {"hr": 35}),
            _proj(2, "pitcher", {"k": 200}),
        ]
        # Add league pool
        projections += [_proj(i, "batter", {"hr": 20}) for i in range(10, 178)]
        projections += [_proj(i, "pitcher", {"k": 150}) for i in range(200, 320)]
        result = analyze_roster([1, 2], projections, league)
        categories = [p.category for p in result.projections]
        assert "hr" in categories
        assert "k" in categories
        assert len(result.projections) == 2


class TestEdgeCases:
    def test_empty_roster(self) -> None:
        """Empty roster → empty analysis."""
        league = _league(batting=(HR_CAT,))
        projections = [_proj(1, "batter", {"hr": 30})]
        result = analyze_roster([], projections, league)
        assert result.projections == []
        assert result.strongest_categories == []
        assert result.weakest_categories == []

    def test_player_not_in_projections(self) -> None:
        """Players without projections are skipped."""
        league = _league(batting=(HR_CAT,))
        projections = [_proj(1, "batter", {"hr": 30})]
        # Player 999 has no projection
        result = analyze_roster([1, 999], projections, league)
        assert result.projections[0].projected_value == pytest.approx(30.0)


class TestStrongestWeakest:
    def test_populated_correctly(self) -> None:
        """strongest_categories and weakest_categories reflect the actual analysis."""
        league = _league(
            teams=12,
            batting=(HR_CAT, SB_CAT),
        )
        # League pool: 168 batters (14 per team × 12 teams)
        projections = [_proj(i, "batter", {"hr": 20, "sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Our roster: strong HR, weak SB
        projections += [_proj(pid, "batter", {"hr": 35, "sb": 5}) for pid in roster_ids]
        result = analyze_roster(roster_ids, projections, league)
        assert "hr" in result.strongest_categories
        assert "sb" in result.weakest_categories
