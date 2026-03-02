from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.pick_value import compute_pick_value_curve, value_at


def _league(teams: int = 4, roster_batters: int = 3, roster_pitchers: int = 2) -> LeagueSettings:
    batting_cats = tuple(
        CategoryConfig(key=k, name=k.upper(), stat_type=StatType.COUNTING, direction=Direction.HIGHER)
        for k in ("hr", "r", "rbi")
    )
    pitching_cats = tuple(
        CategoryConfig(key=k, name=k.upper(), stat_type=StatType.COUNTING, direction=Direction.HIGHER)
        for k in ("w", "sv")
    )
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=teams,
        budget=260,
        roster_batters=roster_batters,
        roster_pitchers=roster_pitchers,
        batting_categories=batting_cats,
        pitching_categories=pitching_cats,
    )


def _adp(player_id: int, overall_pick: float, rank: int | None = None) -> ADP:
    return ADP(
        player_id=player_id,
        season=2026,
        provider="yahoo",
        overall_pick=overall_pick,
        rank=rank if rank is not None else int(overall_pick),
        positions="OF",
    )


def _valuation(player_id: int, value: float) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="zar",
        version="v1",
        projection_system="steamer",
        projection_version="v1",
        player_type="batter",
        position="OF",
        value=value,
        rank=1,
        category_scores={},
    )


class TestComputePickValueCurve:
    def test_basic_join(self) -> None:
        """Players with ADP and valuations produce correct curve values."""
        adp = [_adp(1, 1.0), _adp(2, 2.0), _adp(3, 3.0), _adp(4, 4.0), _adp(5, 5.0)]
        vals = [
            _valuation(1, 40.0),
            _valuation(2, 30.0),
            _valuation(3, 20.0),
            _valuation(4, 10.0),
            _valuation(5, 5.0),
        ]
        league = _league(teams=1, roster_batters=5, roster_pitchers=0)
        curve = compute_pick_value_curve(adp, vals, league)

        # Pick 1 should have highest value, pick 5 the lowest
        assert value_at(curve, 1) > value_at(curve, 5)
        assert curve.total_picks == 5
        assert len(curve.picks) == 5

    def test_monotonically_non_increasing(self) -> None:
        """Curve values never increase as pick number goes up."""
        adp = [_adp(i, float(i)) for i in range(1, 21)]
        # Deliberately noisy values
        raw_values = [50, 45, 48, 40, 35, 38, 30, 25, 28, 20, 15, 18, 12, 10, 8, 6, 5, 4, 3, 2]
        vals = [_valuation(i, float(raw_values[i - 1])) for i in range(1, 21)]
        league = _league(teams=1, roster_batters=20, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league)

        for i in range(1, curve.total_picks):
            assert value_at(curve, i) >= value_at(curve, i + 1), (
                f"Pick {i} ({value_at(curve, i)}) < pick {i + 1} ({value_at(curve, i + 1)})"
            )

    def test_gaps_interpolated(self) -> None:
        """Picks without direct ADP data get interpolated values."""
        adp = [_adp(1, 1.0), _adp(2, 2.0), _adp(5, 5.0)]
        vals = [_valuation(1, 40.0), _valuation(2, 30.0), _valuation(5, 10.0)]
        league = _league(teams=1, roster_batters=5, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league)

        # Picks 3 and 4 should have interpolated values between pick 2 and pick 5
        assert value_at(curve, 3) > 0.0
        assert value_at(curve, 4) > 0.0
        assert value_at(curve, 2) >= value_at(curve, 3) >= value_at(curve, 4) >= value_at(curve, 5)

    def test_missing_valuation_skipped(self) -> None:
        """Player in ADP but not in valuations is gracefully skipped."""
        adp = [_adp(1, 1.0), _adp(2, 2.0), _adp(99, 3.0)]  # player 99 has no valuation
        vals = [_valuation(1, 40.0), _valuation(2, 30.0)]
        league = _league(teams=1, roster_batters=3, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league)

        # Should not crash; curve should still cover all picks
        assert curve.total_picks == 3
        assert len(curve.picks) == 3

    def test_valuation_without_adp_ignored(self) -> None:
        """Player in valuations but not in ADP is ignored."""
        adp = [_adp(1, 1.0), _adp(2, 2.0)]
        vals = [_valuation(1, 40.0), _valuation(2, 30.0), _valuation(99, 100.0)]
        league = _league(teams=1, roster_batters=3, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league)

        # Player 99's high value shouldn't appear at any pick
        for pv in curve.picks:
            assert pv.expected_value <= 40.0

    def test_covers_full_draft(self) -> None:
        """Curve covers teams * (roster_batters + roster_pitchers) picks."""
        adp = [_adp(i, float(i)) for i in range(1, 11)]
        vals = [_valuation(i, float(50 - i * 3)) for i in range(1, 11)]
        league = _league(teams=4, roster_batters=3, roster_pitchers=2)

        curve = compute_pick_value_curve(adp, vals, league)

        expected_total = 4 * (3 + 2)  # 20
        assert curve.total_picks == expected_total
        assert len(curve.picks) == expected_total

    def test_confidence_assignment(self) -> None:
        """Direct matches get 'high', interpolated get 'medium', extrapolated get 'low'."""
        adp = [_adp(1, 1.0), _adp(2, 5.0)]
        vals = [_valuation(1, 40.0), _valuation(2, 20.0)]
        league = _league(teams=1, roster_batters=10, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league)

        # Pick 1 has direct data → high
        pick1 = next(pv for pv in curve.picks if pv.pick == 1)
        assert pick1.confidence == "high"

        # Pick 5 has direct data → high
        pick5 = next(pv for pv in curve.picks if pv.pick == 5)
        assert pick5.confidence == "high"

        # Picks 2-4 are interpolated → medium
        for pick_num in (2, 3, 4):
            pv = next(p for p in curve.picks if p.pick == pick_num)
            assert pv.confidence == "medium", f"Pick {pick_num} should be medium, got {pv.confidence}"

        # Picks 6-10 are extrapolated → low
        for pick_num in (6, 7, 8, 9, 10):
            pv = next(p for p in curve.picks if p.pick == pick_num)
            assert pv.confidence == "low", f"Pick {pick_num} should be low, got {pv.confidence}"

    def test_player_names_included(self) -> None:
        """Player names are attached to picks when provided."""
        adp = [_adp(1, 1.0), _adp(2, 2.0)]
        vals = [_valuation(1, 40.0), _valuation(2, 30.0)]
        names = {1: "Mike Trout", 2: "Shohei Ohtani"}
        league = _league(teams=1, roster_batters=2, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league, player_names=names)

        pick1 = next(pv for pv in curve.picks if pv.pick == 1)
        assert pick1.player_name == "Mike Trout"
        pick2 = next(pv for pv in curve.picks if pv.pick == 2)
        assert pick2.player_name == "Shohei Ohtani"

    def test_season_and_metadata(self) -> None:
        """Curve captures season, provider, and system from inputs."""
        adp = [_adp(1, 1.0)]
        vals = [_valuation(1, 40.0)]
        league = _league(teams=1, roster_batters=1, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league)

        assert curve.season == 2026
        assert curve.provider == "yahoo"
        assert curve.system == "zar"

    def test_multiple_players_same_pick_averaged(self) -> None:
        """When multiple players map to the same pick, values are averaged."""
        adp = [_adp(1, 1.0), _adp(2, 1.0), _adp(3, 2.0)]  # players 1 & 2 both at pick 1
        vals = [_valuation(1, 40.0), _valuation(2, 20.0), _valuation(3, 10.0)]
        league = _league(teams=1, roster_batters=2, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league)

        # Pick 1 should be near the average of 40 and 20 (30), after smoothing
        pick1 = next(pv for pv in curve.picks if pv.pick == 1)
        assert pick1.expected_value > 15.0  # at least higher than pick 2's player
