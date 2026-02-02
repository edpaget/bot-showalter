from fantasy_baseball_manager.draft.rules import (
    MaxPositionCount,
    PitcherBatterRatio,
    PositionNotBeforeRound,
)
from fantasy_baseball_manager.draft.simulation_models import DraftRule, SimulationPick
from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue, StatCategory


def _make_player(
    player_id: str = "p1",
    name: str = "Player 1",
    position_type: str = "B",
) -> PlayerValue:
    return PlayerValue(
        player_id=player_id,
        name=name,
        category_values=(CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=2.0),),
        total_value=2.0,
        position_type=position_type,
    )


def _make_pick(
    team_id: int = 1,
    player_id: str = "px",
    position: str | None = "OF",
    overall_pick: int = 1,
) -> SimulationPick:
    return SimulationPick(
        overall_pick=overall_pick,
        round_number=1,
        pick_in_round=1,
        team_id=team_id,
        team_name="Team 1",
        player_id=player_id,
        player_name="Player X",
        position=position,
        adjusted_value=5.0,
    )


class TestPositionNotBeforeRound:
    def test_implements_draft_rule_protocol(self) -> None:
        rule = PositionNotBeforeRound(position="RP", earliest_round=10)
        assert isinstance(rule, DraftRule)

    def test_has_name(self) -> None:
        rule = PositionNotBeforeRound(position="RP", earliest_round=10)
        assert rule.name

    def test_vetoes_position_before_earliest_round(self) -> None:
        rule = PositionNotBeforeRound(position="RP", earliest_round=10)
        player = _make_player(position_type="P")
        result = rule.evaluate(
            player=player,
            eligible_positions=("RP",),
            round_number=5,
            total_rounds=20,
            picks_so_far=[],
        )
        assert result == 0.0

    def test_allows_position_at_earliest_round(self) -> None:
        rule = PositionNotBeforeRound(position="RP", earliest_round=10)
        player = _make_player(position_type="P")
        result = rule.evaluate(
            player=player,
            eligible_positions=("RP",),
            round_number=10,
            total_rounds=20,
            picks_so_far=[],
        )
        assert result == 1.0

    def test_allows_position_after_earliest_round(self) -> None:
        rule = PositionNotBeforeRound(position="RP", earliest_round=10)
        player = _make_player(position_type="P")
        result = rule.evaluate(
            player=player,
            eligible_positions=("RP",),
            round_number=15,
            total_rounds=20,
            picks_so_far=[],
        )
        assert result == 1.0

    def test_allows_different_position(self) -> None:
        rule = PositionNotBeforeRound(position="RP", earliest_round=10)
        player = _make_player(position_type="P")
        result = rule.evaluate(
            player=player,
            eligible_positions=("SP",),
            round_number=1,
            total_rounds=20,
            picks_so_far=[],
        )
        assert result == 1.0

    def test_vetoes_when_position_is_among_multiple(self) -> None:
        rule = PositionNotBeforeRound(position="RP", earliest_round=10)
        player = _make_player(position_type="P")
        # Player is eligible for both SP and RP — but only has RP
        result = rule.evaluate(
            player=player,
            eligible_positions=("RP",),
            round_number=5,
            total_rounds=20,
            picks_so_far=[],
        )
        assert result == 0.0


class TestMaxPositionCount:
    def test_implements_draft_rule_protocol(self) -> None:
        rule = MaxPositionCount(position="C", max_count=2)
        assert isinstance(rule, DraftRule)

    def test_has_name(self) -> None:
        rule = MaxPositionCount(position="C", max_count=2)
        assert rule.name

    def test_allows_when_under_max(self) -> None:
        rule = MaxPositionCount(position="C", max_count=2)
        player = _make_player()
        picks = [_make_pick(position="C")]
        result = rule.evaluate(
            player=player,
            eligible_positions=("C",),
            round_number=5,
            total_rounds=20,
            picks_so_far=picks,
        )
        assert result == 1.0

    def test_vetoes_when_at_max(self) -> None:
        rule = MaxPositionCount(position="C", max_count=2)
        player = _make_player()
        picks = [_make_pick(position="C"), _make_pick(position="C")]
        result = rule.evaluate(
            player=player,
            eligible_positions=("C",),
            round_number=5,
            total_rounds=20,
            picks_so_far=picks,
        )
        assert result == 0.0

    def test_allows_different_position(self) -> None:
        rule = MaxPositionCount(position="C", max_count=2)
        player = _make_player()
        picks = [_make_pick(position="C"), _make_pick(position="C")]
        result = rule.evaluate(
            player=player,
            eligible_positions=("1B",),
            round_number=5,
            total_rounds=20,
            picks_so_far=picks,
        )
        assert result == 1.0

    def test_allows_when_no_prior_picks(self) -> None:
        rule = MaxPositionCount(position="C", max_count=2)
        player = _make_player()
        result = rule.evaluate(
            player=player,
            eligible_positions=("C",),
            round_number=1,
            total_rounds=20,
            picks_so_far=[],
        )
        assert result == 1.0


class TestPitcherBatterRatio:
    def test_implements_draft_rule_protocol(self) -> None:
        rule = PitcherBatterRatio(max_pitcher_fraction=0.6)
        assert isinstance(rule, DraftRule)

    def test_has_name(self) -> None:
        rule = PitcherBatterRatio(max_pitcher_fraction=0.6)
        assert rule.name

    def test_allows_when_no_prior_picks(self) -> None:
        rule = PitcherBatterRatio(max_pitcher_fraction=0.6)
        player = _make_player(position_type="P")
        result = rule.evaluate(
            player=player,
            eligible_positions=("SP",),
            round_number=1,
            total_rounds=20,
            picks_so_far=[],
        )
        assert result == 1.0

    def test_allows_batter_regardless(self) -> None:
        rule = PitcherBatterRatio(max_pitcher_fraction=0.5)
        player = _make_player(position_type="B")
        # 5 pitchers, 5 batters = 50% pitchers, at max
        picks = [_make_pick(position="SP")] * 5 + [_make_pick(position="OF")] * 5
        result = rule.evaluate(
            player=player,
            eligible_positions=("OF",),
            round_number=11,
            total_rounds=20,
            picks_so_far=picks,
        )
        assert result == 1.0

    def test_penalizes_when_pitcher_fraction_exceeded(self) -> None:
        rule = PitcherBatterRatio(max_pitcher_fraction=0.5)
        player = _make_player(position_type="P")
        # 6 pitchers, 4 batters = 60% pitchers, over 50% max
        picks = [_make_pick(position="SP")] * 6 + [_make_pick(position="OF")] * 4
        result = rule.evaluate(
            player=player,
            eligible_positions=("SP",),
            round_number=11,
            total_rounds=20,
            picks_so_far=picks,
        )
        assert 0.0 < result < 1.0

    def test_allows_pitcher_when_under_fraction(self) -> None:
        rule = PitcherBatterRatio(max_pitcher_fraction=0.6)
        player = _make_player(position_type="P")
        # 2 pitchers, 8 batters = 20% pitchers, well under 60%
        picks = [_make_pick(position="SP")] * 2 + [_make_pick(position="OF")] * 8
        result = rule.evaluate(
            player=player,
            eligible_positions=("SP",),
            round_number=11,
            total_rounds=20,
            picks_so_far=picks,
        )
        assert result == 1.0
