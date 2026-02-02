from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot
from fantasy_baseball_manager.draft.state import FILLED_POSITION_MULTIPLIER, DraftState
from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue, StatCategory


def _make_player_value(
    player_id: str,
    name: str,
    hr_value: float = 1.0,
    sb_value: float = 0.5,
    position_type: str = "B",
) -> PlayerValue:
    return PlayerValue(
        player_id=player_id,
        name=name,
        category_values=(
            CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=hr_value),
            CategoryValue(category=StatCategory.SB, raw_stat=15.0, value=sb_value),
        ),
        total_value=hr_value + sb_value,
        position_type=position_type,
    )


def _simple_roster() -> RosterConfig:
    return RosterConfig(
        slots=(
            RosterSlot(position="1B", count=1),
            RosterSlot(position="OF", count=2),
            RosterSlot(position="Util", count=1),
            RosterSlot(position="SP", count=2),
            RosterSlot(position="BN", count=1),
        )
    )


def _make_state(
    players: list[PlayerValue] | None = None,
    positions: dict[tuple[str, str], tuple[str, ...]] | None = None,
    weights: dict[StatCategory, float] | None = None,
    roster: RosterConfig | None = None,
) -> DraftState:
    if players is None:
        players = [
            _make_player_value("b1", "Batter One", hr_value=2.0, sb_value=1.0),
            _make_player_value("b2", "Batter Two", hr_value=1.0, sb_value=2.0),
            _make_player_value("b3", "Batter Three", hr_value=0.5, sb_value=0.5),
        ]
    return DraftState(
        roster_config=roster or _simple_roster(),
        player_values=players,
        player_positions=positions or {},
        category_weights=weights or {},
    )


class TestGetRankings:
    def test_returns_all_undrafted_players(self) -> None:
        state = _make_state()
        rankings = state.get_rankings()
        assert len(rankings) == 3

    def test_sorted_by_adjusted_value_descending(self) -> None:
        state = _make_state()
        rankings = state.get_rankings()
        values = [r.adjusted_value for r in rankings]
        assert values == sorted(values, reverse=True)

    def test_rank_numbers_are_sequential(self) -> None:
        state = _make_state()
        rankings = state.get_rankings()
        assert [r.rank for r in rankings] == [1, 2, 3]

    def test_limit_restricts_output(self) -> None:
        state = _make_state()
        rankings = state.get_rankings(limit=2)
        assert len(rankings) == 2
        assert rankings[0].rank == 1
        assert rankings[1].rank == 2

    def test_raw_value_is_unweighted_sum(self) -> None:
        state = _make_state()
        rankings = state.get_rankings()
        first = rankings[0]
        assert first.raw_value == sum(cv.value for cv in first.category_values)


class TestCategoryWeighting:
    def test_weight_multiplies_category_value(self) -> None:
        state = _make_state(weights={StatCategory.HR: 2.0})
        rankings = state.get_rankings()
        # Batter One: HR=2.0*2.0 + SB=1.0*1.0 = 5.0
        # Batter Two: HR=1.0*2.0 + SB=2.0*1.0 = 4.0
        first = rankings[0]
        assert first.player_id == "b1"
        assert first.weighted_value == 5.0

    def test_weight_changes_ranking_order(self) -> None:
        # Without weight: b1=3.0, b2=3.0 (tie)
        # With SB weight=3.0: b1=2.0+3.0=5.0, b2=1.0+6.0=7.0
        state = _make_state(weights={StatCategory.SB: 3.0})
        rankings = state.get_rankings()
        assert rankings[0].player_id == "b2"

    def test_default_weight_is_one(self) -> None:
        state = _make_state(weights={})
        rankings = state.get_rankings()
        first = rankings[0]
        # No weights specified, so all categories use 1.0
        assert first.weighted_value == first.raw_value


class TestDraftExclusion:
    def test_drafted_player_excluded_from_rankings(self) -> None:
        state = _make_state()
        state.draft_player("b1", is_user=False)
        rankings = state.get_rankings()
        ids = [r.player_id for r in rankings]
        assert "b1" not in ids
        assert len(rankings) == 2

    def test_multiple_drafts_exclude_multiple(self) -> None:
        state = _make_state()
        state.draft_player("b1", is_user=False)
        state.draft_player("b2", is_user=True, position="OF")
        rankings = state.get_rankings()
        assert len(rankings) == 1
        assert rankings[0].player_id == "b3"


class TestPositionMultiplier:
    def test_no_positions_gives_multiplier_one(self) -> None:
        state = _make_state(positions={})
        rankings = state.get_rankings()
        for r in rankings:
            assert r.position_multiplier == 1.0

    def test_unfilled_position_gives_multiplier_one(self) -> None:
        state = _make_state(
            positions={("b1", "B"): ("1B",), ("b2", "B"): ("OF",), ("b3", "B"): ("OF",)},
        )
        rankings = state.get_rankings()
        for r in rankings:
            assert r.position_multiplier == 1.0

    def test_filled_position_gives_penalty(self) -> None:
        roster = RosterConfig(slots=(RosterSlot(position="1B", count=1),))
        players = [
            _make_player_value("b1", "Batter One", hr_value=2.0, sb_value=1.0),
            _make_player_value("b2", "Batter Two", hr_value=1.5, sb_value=1.0),
        ]
        state = DraftState(
            roster_config=roster,
            player_values=players,
            player_positions={("b1", "B"): ("1B",), ("b2", "B"): ("1B",)},
            category_weights={},
        )
        state.draft_player("b1", is_user=True, position="1B")
        rankings = state.get_rankings()
        b2 = rankings[0]
        assert b2.position_multiplier == FILLED_POSITION_MULTIPLIER

    def test_best_position_picks_most_remaining(self) -> None:
        roster = RosterConfig(
            slots=(
                RosterSlot(position="1B", count=1),
                RosterSlot(position="OF", count=3),
            )
        )
        players = [_make_player_value("b1", "Batter One")]
        state = DraftState(
            roster_config=roster,
            player_values=players,
            player_positions={("b1", "B"): ("1B", "OF")},
            category_weights={},
        )
        rankings = state.get_rankings()
        assert rankings[0].best_position == "OF"

    def test_eligible_position_preferred_over_util_fallback(self) -> None:
        roster = RosterConfig(
            slots=(
                RosterSlot(position="1B", count=1),
                RosterSlot(position="Util", count=2),
            )
        )
        players = [_make_player_value("b1", "Batter One")]
        state = DraftState(
            roster_config=roster,
            player_values=players,
            player_positions={("b1", "B"): ("1B",)},
            category_weights={},
        )
        rankings = state.get_rankings()
        # 1B has remaining capacity, so it should be preferred over Util fallback
        assert rankings[0].best_position == "1B"

    def test_util_used_as_fallback_when_eligible_filled(self) -> None:
        roster = RosterConfig(
            slots=(
                RosterSlot(position="1B", count=1),
                RosterSlot(position="Util", count=1),
            )
        )
        players = [
            _make_player_value("b1", "Batter One", hr_value=2.0),
            _make_player_value("b2", "Batter Two", hr_value=1.0),
        ]
        state = DraftState(
            roster_config=roster,
            player_values=players,
            player_positions={("b1", "B"): ("1B",), ("b2", "B"): ("1B",)},
            category_weights={},
        )
        state.draft_player("b1", is_user=True, position="1B")
        rankings = state.get_rankings()
        # 1B is filled, so Util should be the fallback
        assert rankings[0].best_position == "Util"
        assert rankings[0].position_multiplier == 1.0

    def test_bn_used_only_when_eligible_and_util_filled(self) -> None:
        roster = RosterConfig(
            slots=(
                RosterSlot(position="1B", count=1),
                RosterSlot(position="Util", count=1),
                RosterSlot(position="BN", count=2),
            )
        )
        players = [
            _make_player_value("b1", "One", hr_value=3.0),
            _make_player_value("b2", "Two", hr_value=2.0),
            _make_player_value("b3", "Three", hr_value=1.0),
        ]
        state = DraftState(
            roster_config=roster,
            player_values=players,
            player_positions={
                ("b1", "B"): ("1B",),
                ("b2", "B"): ("1B",),
                ("b3", "B"): ("1B",),
            },
            category_weights={},
        )
        state.draft_player("b1", is_user=True, position="1B")
        state.draft_player("b2", is_user=True, position="Util")
        rankings = state.get_rankings()
        assert rankings[0].best_position == "BN"
        assert rankings[0].position_multiplier == 1.0

    def test_position_penalty_affects_adjusted_value(self) -> None:
        roster = RosterConfig(slots=(RosterSlot(position="1B", count=1),))
        players = [
            _make_player_value("b1", "Strong", hr_value=3.0, sb_value=0.0),
            _make_player_value("b2", "Decent", hr_value=2.0, sb_value=0.0),
        ]
        state = DraftState(
            roster_config=roster,
            player_values=players,
            player_positions={("b1", "B"): ("1B",), ("b2", "B"): ("1B",)},
            category_weights={},
        )
        # Fill the 1B slot
        state.draft_player("b2", is_user=True, position="1B")
        rankings = state.get_rankings()
        b1 = rankings[0]
        assert b1.adjusted_value == b1.weighted_value * FILLED_POSITION_MULTIPLIER


class TestPositionNeeds:
    def test_initial_needs_match_config(self) -> None:
        state = _make_state()
        needs = state.position_needs()
        assert needs["1B"] == 1
        assert needs["OF"] == 2
        assert needs["SP"] == 2

    def test_drafting_reduces_need(self) -> None:
        state = _make_state(
            positions={("b1", "B"): ("OF",)},
        )
        state.draft_player("b1", is_user=True, position="OF")
        needs = state.position_needs()
        assert needs["OF"] == 1

    def test_need_floors_at_zero(self) -> None:
        state = _make_state(
            positions={("b1", "B"): ("1B",)},
        )
        state.draft_player("b1", is_user=True, position="1B")
        # Draft more than capacity
        state.draft_player("b2", is_user=True, position="1B")
        needs = state.position_needs()
        assert needs["1B"] == 0

    def test_opponent_draft_does_not_affect_needs(self) -> None:
        state = _make_state()
        state.draft_player("b1", is_user=False)
        needs = state.position_needs()
        assert needs["1B"] == 1


class TestTwoWayPlayer:
    def test_both_entries_appear_in_rankings(self) -> None:
        players = [
            _make_player_value("ohtani", "Shohei Ohtani", hr_value=3.0, sb_value=1.0, position_type="B"),
            _make_player_value("ohtani", "Shohei Ohtani", hr_value=2.0, sb_value=0.5, position_type="P"),
        ]
        state = _make_state(
            players=players,
            positions={("ohtani", "B"): ("OF",), ("ohtani", "P"): ("SP",)},
        )
        rankings = state.get_rankings()
        assert len(rankings) == 2
        assert all(r.player_id == "ohtani" for r in rankings)
        # Batting entry (higher value) should rank first
        assert rankings[0].eligible_positions == ("OF",)
        assert rankings[1].eligible_positions == ("SP",)

    def test_draft_player_removes_both_entries(self) -> None:
        players = [
            _make_player_value("ohtani", "Shohei Ohtani", hr_value=3.0, sb_value=1.0, position_type="B"),
            _make_player_value("ohtani", "Shohei Ohtani", hr_value=2.0, sb_value=0.5, position_type="P"),
            _make_player_value("other", "Other Player", hr_value=1.0, sb_value=0.5, position_type="B"),
        ]
        state = _make_state(
            players=players,
            positions={("ohtani", "B"): ("OF",), ("ohtani", "P"): ("SP",), ("other", "B"): ("1B",)},
        )
        state.draft_player("ohtani", is_user=False)
        rankings = state.get_rankings()
        assert len(rankings) == 1
        assert rankings[0].player_id == "other"

    def test_draft_player_returns_name(self) -> None:
        players = [
            _make_player_value("ohtani", "Shohei Ohtani", position_type="B"),
            _make_player_value("ohtani", "Shohei Ohtani", position_type="P"),
        ]
        state = _make_state(players=players)
        pick = state.draft_player("ohtani", is_user=False)
        assert pick.name == "Shohei Ohtani"
