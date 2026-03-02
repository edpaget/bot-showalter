import pytest

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.services.category_tracker import (
    analyze_roster,
    compute_category_balance_scores,
    identify_needs,
)


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


class TestIdentifyNeeds:
    """Tests for identify_needs()."""

    def test_weak_category_identified_strong_excluded(self) -> None:
        """Weak categories appear in result; strong/average do not."""
        league = _league(teams=12, batting=(HR_CAT, SB_CAT))
        # League pool: 168 batters with avg stats
        projections = [_proj(i, "batter", {"hr": 20, "sb": 15}) for i in range(1, 169)]
        # Roster: strong HR, weak SB
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"hr": 35, "sb": 5}) for pid in roster_ids]
        # Available player who helps SB
        projections.append(_proj(300, "batter", {"hr": 10, "sb": 30}))

        needs = identify_needs(roster_ids, [300], projections, league)
        categories = [n.category for n in needs]
        assert "sb" in categories
        assert "hr" not in categories

    def test_no_weak_categories_empty_result(self) -> None:
        """When all categories are strong/average, result is empty."""
        league = _league(teams=12, batting=(HR_CAT,))
        projections = [_proj(i, "batter", {"hr": 20}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Strong HR
        projections += [_proj(pid, "batter", {"hr": 35}) for pid in roster_ids]

        needs = identify_needs(roster_ids, [300], projections, league)
        assert needs == []

    def test_best_available_sorted_by_impact_limited_to_top_n(self) -> None:
        """Best available sorted by impact descending, limited to top_n."""
        league = _league(teams=12, batting=(SB_CAT,))
        projections = [_proj(i, "batter", {"sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"sb": 5}) for pid in roster_ids]
        # 7 available players with varying SB
        available_ids = list(range(300, 307))
        for i, aid in enumerate(available_ids):
            projections.append(_proj(aid, "batter", {"sb": 10.0 + i * 5}))

        needs = identify_needs(roster_ids, available_ids, projections, league, top_n=3)
        sb_need = needs[0]
        assert len(sb_need.best_available) == 3
        # Highest impact first
        assert sb_need.best_available[0].category_impact >= sb_need.best_available[1].category_impact
        assert sb_need.best_available[1].category_impact >= sb_need.best_available[2].category_impact

    def test_counting_stat_impact_is_player_value(self) -> None:
        """For counting stats, impact is the player's stat value."""
        league = _league(teams=12, batting=(SB_CAT,))
        projections = [_proj(i, "batter", {"sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"sb": 5}) for pid in roster_ids]
        projections.append(_proj(300, "batter", {"sb": 25}))

        needs = identify_needs(roster_ids, [300], projections, league)
        sb_need = needs[0]
        # Impact for counting stat = player's SB value
        assert sb_need.best_available[0].category_impact == pytest.approx(25.0)

    def test_rate_stat_low_avg_player_negative_impact(self) -> None:
        """A low-AVG player hurts roster AVG (negative impact)."""
        league = _league(teams=12, batting=(AVG_CAT,))
        # League pool: 168 batters with .260 avg (130 H / 500 AB)
        projections = [_proj(i, "batter", {"h": 130, "ab": 500}) for i in range(1, 169)]
        # Roster: 14 batters with low AVG (.200) → weak
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"h": 100, "ab": 500}) for pid in roster_ids]
        # Available player with even worse AVG (.150)
        projections.append(_proj(300, "batter", {"h": 75, "ab": 500}))

        needs = identify_needs(roster_ids, [300], projections, league)
        avg_need = [n for n in needs if n.category == "avg"][0]
        # Low-AVG player should have negative impact (hurts the roster)
        assert avg_need.best_available[0].category_impact < 0

    def test_tradeoff_detected_helps_sb_hurts_avg(self) -> None:
        """When both SB and AVG are weak, a high-SB low-AVG player flags AVG tradeoff."""
        league = _league(teams=12, batting=(SB_CAT, AVG_CAT))
        projections = [_proj(i, "batter", {"sb": 15, "h": 140, "ab": 500}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Weak in both SB and AVG
        projections += [_proj(pid, "batter", {"sb": 5, "h": 100, "ab": 500}) for pid in roster_ids]
        # Available: great SB but terrible AVG
        projections.append(_proj(300, "batter", {"sb": 40, "h": 50, "ab": 500}))

        needs = identify_needs(roster_ids, [300], projections, league)
        sb_need = [n for n in needs if n.category == "sb"][0]
        rec = sb_need.best_available[0]
        assert "avg" in rec.tradeoff_categories

    def test_tradeoff_not_flagged_for_strong_category(self) -> None:
        """Worsening a strong category does not produce a tradeoff warning."""
        league = _league(teams=12, batting=(SB_CAT, HR_CAT))
        projections = [_proj(i, "batter", {"sb": 15, "hr": 20}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Weak SB, strong HR
        projections += [_proj(pid, "batter", {"sb": 5, "hr": 35}) for pid in roster_ids]
        # Available: good SB, low HR (would hurt HR, but HR is strong)
        projections.append(_proj(300, "batter", {"sb": 30, "hr": 5}))

        needs = identify_needs(roster_ids, [300], projections, league)
        sb_need = [n for n in needs if n.category == "sb"][0]
        rec = sb_need.best_available[0]
        # HR is strong, so not a tradeoff
        assert "hr" not in rec.tradeoff_categories

    def test_counting_stat_no_tradeoff(self) -> None:
        """Counting stats can't have negative impact (sum never decreases)."""
        league = _league(teams=12, batting=(HR_CAT, SB_CAT))
        projections = [_proj(i, "batter", {"hr": 20, "sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Weak in both
        projections += [_proj(pid, "batter", {"hr": 10, "sb": 5}) for pid in roster_ids]
        # Available: helps HR, has no SB
        projections.append(_proj(300, "batter", {"hr": 25, "sb": 0}))

        needs = identify_needs(roster_ids, [300], projections, league)
        hr_need = [n for n in needs if n.category == "hr"][0]
        rec = hr_need.best_available[0]
        # SB is weak but adding a player can't decrease the SB sum
        assert "sb" not in rec.tradeoff_categories

    def test_partial_roster_works(self) -> None:
        """With only 2 players instead of 14, identify_needs still works."""
        league = _league(teams=12, batting=(SB_CAT,))
        projections = [_proj(i, "batter", {"sb": 15}) for i in range(1, 169)]
        roster_ids = [200, 201]
        projections += [_proj(pid, "batter", {"sb": 3}) for pid in roster_ids]
        projections.append(_proj(300, "batter", {"sb": 20}))

        needs = identify_needs(roster_ids, [300], projections, league)
        assert len(needs) >= 1
        assert needs[0].category == "sb"

    def test_empty_available_pool_empty_best_available(self) -> None:
        """No available players → empty best_available."""
        league = _league(teams=12, batting=(SB_CAT,))
        projections = [_proj(i, "batter", {"sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"sb": 5}) for pid in roster_ids]

        needs = identify_needs(roster_ids, [], projections, league)
        assert len(needs) >= 1
        assert needs[0].best_available == ()

    def test_pitching_category_era_weak_pitchers_recommended(self) -> None:
        """ERA is weak → pitchers recommended (not batters)."""
        league = _league(teams=12, pitching=(ERA_CAT,))
        # League pool: 120 pitchers with ERA ratio 0.2
        projections = [_proj(i, "pitcher", {"er": 40, "ip": 200}) for i in range(1, 121)]
        # Roster: 10 pitchers with bad ERA (ratio 0.3)
        roster_ids = list(range(200, 210))
        projections += [_proj(pid, "pitcher", {"er": 60, "ip": 200}) for pid in roster_ids]
        # Available pitcher with good ERA
        projections.append(_proj(300, "pitcher", {"er": 15, "ip": 200}))
        # Available batter (should not appear for pitching category)
        projections.append(_proj(301, "batter", {"hr": 30}))

        needs = identify_needs(roster_ids, [300, 301], projections, league)
        era_need = needs[0]
        assert era_need.category == "era"
        player_ids = [r.player_id for r in era_need.best_available]
        assert 300 in player_ids
        assert 301 not in player_ids

    def test_lower_is_better_low_era_positive_impact(self) -> None:
        """For LOWER-is-better stats, a pitcher with low ERA has positive impact."""
        league = _league(teams=12, pitching=(ERA_CAT,))
        projections = [_proj(i, "pitcher", {"er": 40, "ip": 200}) for i in range(1, 121)]
        roster_ids = list(range(200, 210))
        projections += [_proj(pid, "pitcher", {"er": 60, "ip": 200}) for pid in roster_ids]
        # Available pitcher with great ERA (low ER)
        projections.append(_proj(300, "pitcher", {"er": 10, "ip": 200}))

        needs = identify_needs(roster_ids, [300], projections, league)
        era_need = needs[0]
        # Good pitcher should have positive impact (lowers ERA)
        assert era_need.best_available[0].category_impact > 0

    def test_player_names_populated(self) -> None:
        """player_names dict populates PlayerRecommendation.player_name."""
        league = _league(teams=12, batting=(SB_CAT,))
        projections = [_proj(i, "batter", {"sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"sb": 5}) for pid in roster_ids]
        projections.append(_proj(300, "batter", {"sb": 25}))

        names = {300: "Rickey Henderson"}
        needs = identify_needs(roster_ids, [300], projections, league, player_names=names)
        rec = needs[0].best_available[0]
        assert rec.player_name == "Rickey Henderson"

    def test_multiple_weak_categories_sorted_by_worst_rank(self) -> None:
        """Multiple weak categories sorted by worst rank first."""
        league = _league(teams=12, batting=(HR_CAT, SB_CAT))
        projections = [_proj(i, "batter", {"hr": 20, "sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Very weak SB (rank ~12), somewhat weak HR (rank ~10)
        projections += [_proj(pid, "batter", {"hr": 13, "sb": 2}) for pid in roster_ids]
        projections.append(_proj(300, "batter", {"hr": 20, "sb": 20}))

        needs = identify_needs(roster_ids, [300], projections, league)
        assert len(needs) >= 2
        # Worst rank first
        assert needs[0].current_rank >= needs[1].current_rank


class TestComputeCategoryBalanceScores:
    """Tests for compute_category_balance_scores()."""

    def test_high_sb_player_scores_high_when_sb_weak(self) -> None:
        """A player with high SB gets a high score when SB is weak."""
        league = _league(teams=12, batting=(HR_CAT, SB_CAT))
        projections = [_proj(i, "batter", {"hr": 20, "sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Strong HR, weak SB
        projections += [_proj(pid, "batter", {"hr": 35, "sb": 5}) for pid in roster_ids]
        # Available: one with high SB, one without
        projections.append(_proj(300, "batter", {"hr": 10, "sb": 30}))
        projections.append(_proj(301, "batter", {"hr": 30, "sb": 0}))

        scores = compute_category_balance_scores(roster_ids, [300, 301], projections, league)
        assert scores[300] > scores[301]

    def test_no_weak_categories_all_scores_zero(self) -> None:
        """When no categories are weak, all scores should be 0.0."""
        league = _league(teams=12, batting=(HR_CAT,))
        projections = [_proj(i, "batter", {"hr": 20}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        # Strong HR
        projections += [_proj(pid, "batter", {"hr": 35}) for pid in roster_ids]
        projections.append(_proj(300, "batter", {"hr": 25}))

        scores = compute_category_balance_scores(roster_ids, [300], projections, league)
        assert scores[300] == 0.0

    def test_scores_normalized_zero_to_one(self) -> None:
        """All scores should be in [0, 1]."""
        league = _league(teams=12, batting=(HR_CAT, SB_CAT))
        projections = [_proj(i, "batter", {"hr": 20, "sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"hr": 10, "sb": 5}) for pid in roster_ids]
        available_ids = list(range(300, 310))
        for i, aid in enumerate(available_ids):
            projections.append(_proj(aid, "batter", {"hr": float(5 + i * 3), "sb": float(i * 4)}))

        scores = compute_category_balance_scores(roster_ids, available_ids, projections, league)
        for score in scores.values():
            assert 0.0 <= score <= 1.0
        # At least one should be 1.0 (the best) if there are weak categories
        assert max(scores.values()) == pytest.approx(1.0)

    def test_empty_available_returns_empty(self) -> None:
        """No available players → empty dict."""
        league = _league(teams=12, batting=(SB_CAT,))
        projections = [_proj(i, "batter", {"sb": 15}) for i in range(1, 169)]
        roster_ids = list(range(200, 214))
        projections += [_proj(pid, "batter", {"sb": 5}) for pid in roster_ids]

        scores = compute_category_balance_scores(roster_ids, [], projections, league)
        assert scores == {}

    def test_pitching_category_weak_scores_pitchers(self) -> None:
        """When a pitching category is weak, pitchers addressing it score higher."""
        league = _league(teams=12, pitching=(K_CAT,))
        projections = [_proj(i, "pitcher", {"k": 150}) for i in range(1, 121)]
        roster_ids = list(range(200, 210))
        projections += [_proj(pid, "pitcher", {"k": 50}) for pid in roster_ids]
        # Available: one high-K pitcher, one low-K pitcher
        projections.append(_proj(300, "pitcher", {"k": 200}))
        projections.append(_proj(301, "pitcher", {"k": 10}))

        scores = compute_category_balance_scores(roster_ids, [300, 301], projections, league)
        assert scores[300] > scores[301]
