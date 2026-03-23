import pytest

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.domain.keeper import KeeperCost
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.pick_value import (
    PickTrade,
    PickValue,
    PickValueCurve,
)
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.keeper_service import compute_surplus
from fantasy_baseball_manager.services.pick_value import (
    _apply_trade,
    _run_greedy_draft,
    _snake_order,
    cascade_analysis,
    compute_pick_value_curve,
    evaluate_pick_trade,
    evaluate_pick_trade_with_context,
    picks_to_dollar_costs,
    round_to_dollar_cost,
    value_at,
)


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

    def test_two_way_player_both_types_contribute(self) -> None:
        """A two-way player's batter and pitcher valuations both contribute at their ADP pick."""
        # Player 1 is a two-way player: batter value 30, pitcher value 20
        # Player 2 is a regular batter: value 25
        adp = [_adp(1, 1.0), _adp(2, 2.0)]
        vals = [
            _valuation(1, 30.0),  # batter
            Valuation(
                player_id=1,
                season=2026,
                system="zar",
                version="v1",
                projection_system="steamer",
                projection_version="v1",
                player_type="pitcher",
                position="SP",
                value=20.0,
                rank=1,
                category_scores={},
            ),
            _valuation(2, 25.0),
        ]
        league = _league(teams=1, roster_batters=2, roster_pitchers=0)

        curve = compute_pick_value_curve(adp, vals, league)

        # Pick 1 raw values should include both 30 and 20 (averaged to 25),
        # which is >= pick 2's single value of 25.
        # After smoothing/monotonicity the exact values may shift,
        # but pick 1 should reflect both contributions.
        pick1 = next(pv for pv in curve.picks if pv.pick == 1)
        pick2 = next(pv for pv in curve.picks if pv.pick == 2)
        assert pick1.expected_value >= pick2.expected_value


def _steep_curve(num_picks: int = 30) -> PickValueCurve:
    """Build a steep curve where pick 1 is very valuable and value drops fast."""
    picks = [
        PickValue(
            pick=i,
            expected_value=round(100.0 / i, 2),  # 100, 50, 33.3, 25, 20, ...
            player_name=f"Player {i}",
            confidence="high",
        )
        for i in range(1, num_picks + 1)
    ]
    return PickValueCurve(season=2026, provider="yahoo", system="zar", picks=picks, total_picks=num_picks)


class TestEvaluatePickTrade:
    def test_symmetric_trade(self) -> None:
        """Trading pick 10 for pick 10 → net_value ≈ 0, recommendation 'even'."""
        curve = _steep_curve()
        trade = PickTrade(gives=[10], receives=[10])
        result = evaluate_pick_trade(trade, curve)
        assert result.net_value == pytest.approx(0.0)
        assert result.recommendation == "even"

    def test_obvious_win(self) -> None:
        """Trading pick 100 for pick 1 → positive net_value, 'accept'."""
        curve = _steep_curve(num_picks=100)
        trade = PickTrade(gives=[100], receives=[1])
        result = evaluate_pick_trade(trade, curve)
        assert result.net_value > 0.0
        assert result.recommendation == "accept"

    def test_obvious_loss(self) -> None:
        """Trading pick 1 for pick 100 → negative net_value, 'reject'."""
        curve = _steep_curve(num_picks=100)
        trade = PickTrade(gives=[1], receives=[100])
        result = evaluate_pick_trade(trade, curve)
        assert result.net_value < 0.0
        assert result.recommendation == "reject"

    def test_multi_pick_steep_curve_loss(self) -> None:
        """Trading pick 1 for picks 20+21 is a loss when curve is steep (AC #1)."""
        curve = _steep_curve()
        trade = PickTrade(gives=[1], receives=[20, 21])
        result = evaluate_pick_trade(trade, curve)
        # Pick 1 = 100.0, picks 20+21 = 5.0 + 4.76 = 9.76
        assert result.net_value < 0.0
        assert result.recommendation == "reject"

    def test_threshold_makes_small_trade_even(self) -> None:
        """Trade with small net_value below threshold → 'even'."""
        curve = _steep_curve()
        # Picks 10 (10.0) vs pick 11 (9.09) → net = -0.91
        trade = PickTrade(gives=[10], receives=[11])
        result = evaluate_pick_trade(trade, curve, threshold=1.0)
        assert abs(result.net_value) < 1.0
        assert result.recommendation == "even"

    def test_net_value_positive_when_receiving_more(self) -> None:
        """net_value is positive when receiving side has more total expected value (AC #3)."""
        curve = _steep_curve()
        trade = PickTrade(gives=[20], receives=[5])
        result = evaluate_pick_trade(trade, curve)
        assert result.receives_value > result.gives_value
        assert result.net_value > 0.0

    def test_detail_lists_correct(self) -> None:
        """gives_detail and receives_detail contain correct PickValue entries."""
        curve = _steep_curve()
        trade = PickTrade(gives=[1, 5], receives=[3, 10])
        result = evaluate_pick_trade(trade, curve)

        assert len(result.gives_detail) == 2
        assert len(result.receives_detail) == 2

        # Verify pick numbers match
        assert result.gives_detail[0].pick == 1
        assert result.gives_detail[1].pick == 5
        assert result.receives_detail[0].pick == 3
        assert result.receives_detail[1].pick == 10

        # Verify values match what the curve has
        assert result.gives_detail[0].expected_value == value_at(curve, 1)
        assert result.gives_detail[1].expected_value == value_at(curve, 5)
        assert result.receives_detail[0].expected_value == value_at(curve, 3)
        assert result.receives_detail[1].expected_value == value_at(curve, 10)


def _board_row(player_id: int, name: str, rank: int, position: str, value: float) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=rank,
        player_type="batter",
        position=position,
        value=value,
        category_z_scores={},
    )


def _test_board() -> DraftBoard:
    """Board with known players at specific ranks/positions."""
    rows = [
        _board_row(1, "Elite SS", rank=3, position="SS", value=50.0),
        _board_row(2, "Elite OF", rank=4, position="OF", value=48.0),
        _board_row(3, "Good OF", rank=5, position="OF", value=45.0),
        _board_row(4, "OK 1B", rank=6, position="1B", value=40.0),
        _board_row(5, "Mid SS", rank=14, position="SS", value=20.0),
        _board_row(6, "Mid OF", rank=15, position="OF", value=18.0),
        _board_row(7, "Late 1B", rank=16, position="1B", value=15.0),
    ]
    return DraftBoard(
        rows=rows,
        batting_categories=("hr", "r", "rbi"),
        pitching_categories=("w", "sv"),
    )


class TestEvaluatePickTradeWithContext:
    def test_without_positional_need_matches_basic(self) -> None:
        """Without positional need, results match basic evaluate_pick_trade()."""
        curve = _steep_curve()
        trade = PickTrade(gives=[5], receives=[10])
        board = _test_board()

        basic = evaluate_pick_trade(trade, curve)
        ctx = evaluate_pick_trade_with_context(trade, curve, board, needed_positions=[])

        assert ctx.net_value == basic.net_value
        assert ctx.recommendation == basic.recommendation

    def test_positional_need_changes_value(self) -> None:
        """With positional need: pick 5 has a needed SS → trading it away costs more (AC #2)."""
        curve = _steep_curve(num_picks=20)
        board = _test_board()
        # Trade pick 5 (where elite SS is ranked) for pick 15
        trade = PickTrade(gives=[5], receives=[15])

        # Without context: just curve values
        basic = evaluate_pick_trade(trade, curve)

        # With context needing SS: pick 5 has Elite SS (value=45.0) which is much more
        # than the curve value at pick 5 (20.0), making the trade worse
        ctx = evaluate_pick_trade_with_context(trade, curve, board, needed_positions=["SS"])

        # The context-aware net should be worse (more negative) because giving away
        # a pick with a needed SS hurts more
        assert ctx.net_value < basic.net_value

    def test_no_matching_position_falls_back_to_curve(self) -> None:
        """When no player at a pick matches needed position, falls back to curve."""
        curve = _steep_curve(num_picks=20)
        board = _test_board()
        # Need a catcher but nobody in the board is a catcher
        trade = PickTrade(gives=[5], receives=[15])

        basic = evaluate_pick_trade(trade, curve)
        ctx = evaluate_pick_trade_with_context(trade, curve, board, needed_positions=["C"])

        # Should fall back to curve values since no catchers exist
        assert ctx.net_value == basic.net_value

    def test_player_specific_detail(self) -> None:
        """gives_detail and receives_detail reflect actual player values."""
        curve = _steep_curve(num_picks=20)
        board = _test_board()
        trade = PickTrade(gives=[5], receives=[15])

        ctx = evaluate_pick_trade_with_context(trade, curve, board, needed_positions=["SS"])

        # Pick 5 should show the Elite SS player (rank=3, within window of pick 5)
        assert ctx.gives_detail[0].player_name == "Elite SS"
        assert ctx.gives_detail[0].expected_value == 50.0

        # Pick 15 should show the Mid SS player
        assert ctx.receives_detail[0].player_name == "Mid SS"
        assert ctx.receives_detail[0].expected_value == 20.0


class TestSnakeOrder:
    def test_4_team_12_pick(self) -> None:
        """4-team snake: picks 1-4 forward, 5-8 reverse, 9-12 forward."""
        order = _snake_order(teams=4, total_picks=12)
        # Round 1: 1→0, 2→1, 3→2, 4→3
        assert order[1] == 0
        assert order[4] == 3
        # Round 2 (reversed): 5→3, 6→2, 7→1, 8→0
        assert order[5] == 3
        assert order[8] == 0
        # Round 3: 9→0, 10→1, 11→2, 12→3
        assert order[9] == 0
        assert order[12] == 3

    def test_covers_all_picks(self) -> None:
        order = _snake_order(teams=3, total_picks=9)
        assert set(order.keys()) == {1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_each_team_gets_equal_picks(self) -> None:
        order = _snake_order(teams=4, total_picks=20)
        per_team = [0] * 4
        for team in order.values():
            per_team[team] += 1
        assert all(count == 5 for count in per_team)


class TestApplyTrade:
    def test_swap_pick_1_and_4(self) -> None:
        """Swap gives=[1] receives=[4] for user team 0 → pick 1→team 3, pick 4→team 0."""
        order = _snake_order(teams=4, total_picks=12)
        trade = PickTrade(gives=[1], receives=[4])
        new_order = _apply_trade(order, trade, user_team_idx=0)
        # User (team 0) no longer has pick 1, now has pick 4
        assert new_order[1] == 3  # partner gets pick 1
        assert new_order[4] == 0  # user gets pick 4

    def test_gives_pick_not_owned_by_user_raises(self) -> None:
        """ValueError if gives pick doesn't belong to user."""
        order = _snake_order(teams=4, total_picks=12)
        trade = PickTrade(gives=[2], receives=[4])  # pick 2 belongs to team 1
        with pytest.raises(ValueError, match="does not belong to user"):
            _apply_trade(order, trade, user_team_idx=0)

    def test_receives_pick_already_owned_raises(self) -> None:
        """ValueError if receives pick already belongs to user."""
        order = _snake_order(teams=4, total_picks=12)
        trade = PickTrade(gives=[1], receives=[9])  # pick 9 also belongs to team 0
        with pytest.raises(ValueError, match="already belongs to user"):
            _apply_trade(order, trade, user_team_idx=0)

    def test_receives_from_multiple_partners_raises(self) -> None:
        """ValueError if receives picks come from multiple partners."""
        order = _snake_order(teams=4, total_picks=12)
        # pick 2→team 1, pick 3→team 2: different owners
        trade = PickTrade(gives=[1, 9], receives=[2, 3])
        with pytest.raises(ValueError, match="multiple teams"):
            _apply_trade(order, trade, user_team_idx=0)


def _cascade_league() -> LeagueSettings:
    """4-team league with 3 batter slots (C/1B/OF) + 2 pitcher slots."""
    batting_cats = tuple(
        CategoryConfig(key=k, name=k.upper(), stat_type=StatType.COUNTING, direction=Direction.HIGHER)
        for k in ("hr", "r", "rbi")
    )
    pitching_cats = tuple(
        CategoryConfig(key=k, name=k.upper(), stat_type=StatType.COUNTING, direction=Direction.HIGHER)
        for k in ("w", "sv")
    )
    return LeagueSettings(
        name="Cascade Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=4,
        budget=260,
        roster_batters=3,
        roster_pitchers=2,
        batting_categories=batting_cats,
        pitching_categories=pitching_cats,
        positions={"C": 1, "1B": 1, "OF": 1},
    )


def _cascade_board() -> DraftBoard:
    """Board with enough players for a 4-team, 5-round draft (20 picks).

    Batters: 4 C, 4 1B, 4 OF (12 batters for 12 batter slots).
    Pitchers: 8 P (for 8 pitcher slots).
    Values are descending so greedy picks are deterministic.
    """
    rows: list[DraftBoardRow] = []
    pid = 1
    # 4 catchers
    for i in range(4):
        rows.append(_board_row(pid, f"C-{i + 1}", rank=pid, position="C", value=50.0 - i * 5))
        pid += 1
    # 4 first basemen
    for i in range(4):
        rows.append(_board_row(pid, f"1B-{i + 1}", rank=pid, position="1B", value=48.0 - i * 5))
        pid += 1
    # 4 outfielders
    for i in range(4):
        rows.append(_board_row(pid, f"OF-{i + 1}", rank=pid, position="OF", value=46.0 - i * 5))
        pid += 1
    # 8 pitchers
    for i in range(8):
        rows.append(
            DraftBoardRow(
                player_id=pid,
                player_name=f"P-{i + 1}",
                rank=pid,
                player_type="pitcher",
                position="SP",
                value=40.0 - i * 3,
                category_z_scores={},
            )
        )
        pid += 1
    return DraftBoard(
        rows=rows,
        batting_categories=("hr", "r", "rbi"),
        pitching_categories=("w", "sv"),
    )


class TestRunGreedyDraft:
    def test_all_picks_assigned(self) -> None:
        """4-team, 5-slot league: all 20 picks assigned, no duplicates."""
        league = _cascade_league()
        board = _cascade_board()
        order = _snake_order(teams=4, total_picks=20)
        rosters = _run_greedy_draft(board, league, order)
        all_player_ids = [p.player_id for picks in rosters.values() for p in picks]
        assert len(all_player_ids) == 20
        assert len(set(all_player_ids)) == 20  # no duplicates

    def test_each_team_gets_5_picks(self) -> None:
        league = _cascade_league()
        board = _cascade_board()
        order = _snake_order(teams=4, total_picks=20)
        rosters = _run_greedy_draft(board, league, order)
        for team_idx in range(4):
            assert len(rosters[team_idx]) == 5

    def test_greedy_picks_highest_value(self) -> None:
        """Team 0 drafts first and should get the highest-value player."""
        league = _cascade_league()
        board = _cascade_board()
        order = _snake_order(teams=4, total_picks=20)
        rosters = _run_greedy_draft(board, league, order)
        # Team 0 picks first; C-1 (value=50) is the highest-value player
        team0_names = [p.player_name for p in rosters[0]]
        assert "C-1" in team0_names

    def test_deterministic(self) -> None:
        """Same inputs produce identical output."""
        league = _cascade_league()
        board = _cascade_board()
        order = _snake_order(teams=4, total_picks=20)
        r1 = _run_greedy_draft(board, league, order)
        r2 = _run_greedy_draft(board, league, order)
        for team_idx in range(4):
            ids1 = [p.player_id for p in r1[team_idx]]
            ids2 = [p.player_id for p in r2[team_idx]]
            assert ids1 == ids2


class TestCascadeAnalysis:
    def test_different_rosters(self) -> None:
        """AC #1: trade pick 1↔4, before/after user rosters differ."""
        league = _cascade_league()
        board = _cascade_board()
        trade = PickTrade(gives=[1], receives=[4])
        result = cascade_analysis(trade, board, league, user_team_idx=0)
        before_ids = {p.player_id for p in result.before.picks}
        after_ids = {p.player_id for p in result.after.picks}
        assert before_ids != after_ids

    def test_directional_consistency(self) -> None:
        """AC #2: value_delta same sign as evaluate_pick_trade() net_value."""
        league = _cascade_league()
        board = _cascade_board()
        # Trade down: give pick 1, receive pick 4
        trade = PickTrade(gives=[1], receives=[4])
        cascade_result = cascade_analysis(trade, board, league, user_team_idx=0)

        # Build a simple curve from the board for comparison
        curve = _steep_curve(num_picks=20)
        eval_result = evaluate_pick_trade(trade, curve)

        # Both should be negative (trading down)
        assert (cascade_result.value_delta < 0) == (eval_result.net_value < 0)

    def test_trading_down_decreases_value(self) -> None:
        """Give pick 1, receive pick 4 → negative delta."""
        league = _cascade_league()
        board = _cascade_board()
        trade = PickTrade(gives=[1], receives=[4])
        result = cascade_analysis(trade, board, league, user_team_idx=0)
        assert result.value_delta < 0
        assert result.recommendation == "reject"

    def test_cascade_affects_other_teams(self) -> None:
        """Other teams' rosters also change when picks are swapped."""
        league = _cascade_league()
        board = _cascade_board()
        trade = PickTrade(gives=[1], receives=[4])

        # Run both drafts manually to compare all teams
        order_before = _snake_order(teams=4, total_picks=20)
        order_after = _apply_trade(order_before, trade, user_team_idx=0)
        rosters_before = _run_greedy_draft(board, league, order_before)
        rosters_after = _run_greedy_draft(board, league, order_after)

        # The trade partner (team 3) should also have different rosters
        partner_before_ids = {p.player_id for p in rosters_before[3]}
        partner_after_ids = {p.player_id for p in rosters_after[3]}
        assert partner_before_ids != partner_after_ids

    def test_works_standalone(self) -> None:
        """AC #3: no run_batch_simulation dependency — import succeeds and runs."""
        league = _cascade_league()
        board = _cascade_board()
        trade = PickTrade(gives=[1], receives=[4])
        # Just verifying it runs without needing run_batch_simulation
        result = cascade_analysis(trade, board, league, user_team_idx=0)
        assert result.before.total_value > 0
        assert result.after.total_value > 0


class TestRoundToDollarCost:
    """Tests for round_to_dollar_cost(): draft round → dollar value translation."""

    def test_round_1_equals_average_of_first_n_picks(self) -> None:
        """4-team league, round 1 = average of picks 1-4."""
        curve = _steep_curve(num_picks=30)
        league = _league(teams=4)
        cost = round_to_dollar_cost(1, league, curve)
        expected = sum(value_at(curve, p) for p in range(1, 5)) / 4
        assert cost == pytest.approx(expected)

    def test_round_1_greater_than_round_10(self) -> None:
        """Earlier rounds cost more than later rounds."""
        curve = _steep_curve(num_picks=50)
        league = _league(teams=4)
        assert round_to_dollar_cost(1, league, curve) > round_to_dollar_cost(10, league, curve)

    def test_monotonically_decreasing(self) -> None:
        """Costs never increase across rounds 1-15."""
        curve = _steep_curve(num_picks=80)
        league = _league(teams=4)
        costs = [round_to_dollar_cost(r, league, curve) for r in range(1, 16)]
        for i in range(len(costs) - 1):
            assert costs[i] >= costs[i + 1], f"Round {i + 1} ({costs[i]}) < round {i + 2} ({costs[i + 1]})"

    def test_round_beyond_curve_returns_floor(self) -> None:
        """Rounds past curve range → $1 floor."""
        curve = _steep_curve(num_picks=10)
        league = _league(teams=4)
        # Round 10 = picks 37-40, well beyond the 10-pick curve
        cost = round_to_dollar_cost(10, league, curve)
        assert cost == 1.0

    def test_partial_round_beyond_curve(self) -> None:
        """If only some picks in a round have data, average available ones (still ≥ $1)."""
        # Curve with 6 picks; 4-team league round 2 = picks 5-8.
        # Only picks 5-6 have data, 7-8 return 0.0 from value_at.
        curve = _steep_curve(num_picks=6)
        league = _league(teams=4)
        cost = round_to_dollar_cost(2, league, curve)
        # Should average picks 5 and 6 (the positive ones), ignoring 7 and 8
        expected = (value_at(curve, 5) + value_at(curve, 6)) / 2
        assert cost == pytest.approx(expected)
        assert cost >= 1.0

    def test_12_team_league_round_1(self) -> None:
        """12-team league, round 1 = average of picks 1-12."""
        curve = _steep_curve(num_picks=30)
        league = _league(teams=12)
        cost = round_to_dollar_cost(1, league, curve)
        expected = sum(value_at(curve, p) for p in range(1, 13)) / 12
        assert cost == pytest.approx(expected)


class TestPicksToDollarCosts:
    """Tests for picks_to_dollar_costs(): batch round→KeeperCost conversion."""

    def test_produces_keeper_cost_records(self) -> None:
        """Returns KeeperCost instances with source='draft_round' and original_round set."""
        curve = _steep_curve(num_picks=30)
        league = _league(teams=4)
        entries = [(100, 1), (200, 3)]
        result = picks_to_dollar_costs(entries, season=2026, league_name="dynasty", league=league, curve=curve)
        assert len(result) == 2
        for kc in result:
            assert isinstance(kc, KeeperCost)
            assert kc.source == "draft_round"
        assert result[0].original_round == 1
        assert result[1].original_round == 3

    def test_player_ids_and_season_match(self) -> None:
        """Fields propagated correctly to KeeperCost."""
        curve = _steep_curve(num_picks=30)
        league = _league(teams=4)
        entries = [(42, 2)]
        result = picks_to_dollar_costs(entries, season=2026, league_name="dynasty", league=league, curve=curve)
        assert result[0].player_id == 42
        assert result[0].season == 2026
        assert result[0].league == "dynasty"

    def test_costs_decrease_with_later_rounds(self) -> None:
        """Round 1 keeper costs more than round 5 keeper."""
        curve = _steep_curve(num_picks=30)
        league = _league(teams=4)
        entries = [(100, 1), (200, 5)]
        result = picks_to_dollar_costs(entries, season=2026, league_name="lg", league=league, curve=curve)
        assert result[0].cost > result[1].cost

    def test_round_beyond_curve_gets_floor(self) -> None:
        """Floor cost for late rounds beyond curve range."""
        curve = _steep_curve(num_picks=10)
        league = _league(teams=4)
        entries = [(100, 20)]  # round 20 = picks 77-80, way beyond 10-pick curve
        result = picks_to_dollar_costs(entries, season=2026, league_name="lg", league=league, curve=curve)
        assert result[0].cost == 1.0

    def test_empty_entries_returns_empty(self) -> None:
        """Empty input → empty output."""
        curve = _steep_curve(num_picks=30)
        league = _league(teams=4)
        result = picks_to_dollar_costs([], season=2026, league_name="lg", league=league, curve=curve)
        assert result == []

    def test_compatible_with_compute_surplus(self) -> None:
        """Integration: output feeds into compute_surplus() without error."""
        curve = _steep_curve(num_picks=30)
        league = _league(teams=4)
        entries = [(1, 3)]
        costs = picks_to_dollar_costs(entries, season=2026, league_name="lg", league=league, curve=curve)
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        valuations = [_valuation(1, 25.0)]
        # Should not raise — KeeperCost is compatible with compute_surplus
        decisions = compute_surplus(costs, valuations, players)
        assert len(decisions) == 1
