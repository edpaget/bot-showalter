from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot
from fantasy_baseball_manager.valuation.models import (
    CategoryValue,
    PlayerValue,
    StatCategory,
)
from fantasy_baseball_manager.valuation.replacement import (
    BATTER_SCARCITY_ORDER,
    PITCHER_SCARCITY_ORDER,
    PositionThreshold,
    ReplacementConfig,
    apply_replacement_adjustment,
    assign_positions,
    compute_replacement_levels,
)


def _make_player(
    player_id: str, total_value: float, position_type: str = "batter"
) -> PlayerValue:
    return PlayerValue(
        player_id=player_id,
        name=f"Player {player_id}",
        category_values=(),
        total_value=total_value,
        position_type=position_type,
    )


class TestDataTypes:
    def test_position_threshold_construction(self) -> None:
        threshold = PositionThreshold(
            position="C",
            roster_spots=24,
            replacement_rank=24,
            replacement_value=1.5,
        )
        assert threshold.position == "C"
        assert threshold.roster_spots == 24
        assert threshold.replacement_rank == 24
        assert threshold.replacement_value == 1.5

    def test_position_threshold_is_frozen(self) -> None:
        threshold = PositionThreshold(
            position="C",
            roster_spots=24,
            replacement_rank=24,
            replacement_value=1.5,
        )
        with pytest.raises(FrozenInstanceError):
            threshold.position = "SS"  # type: ignore[misc]

    def test_replacement_config_construction(self) -> None:
        roster = RosterConfig(slots=(RosterSlot("C", 2),))
        config = ReplacementConfig(team_count=12, roster_config=roster)
        assert config.team_count == 12
        assert config.roster_config is roster
        assert config.smoothing_window == 5

    def test_replacement_config_custom_smoothing_window(self) -> None:
        roster = RosterConfig(slots=(RosterSlot("C", 2),))
        config = ReplacementConfig(
            team_count=12, roster_config=roster, smoothing_window=3
        )
        assert config.smoothing_window == 3

    def test_replacement_config_is_frozen(self) -> None:
        roster = RosterConfig(slots=(RosterSlot("C", 2),))
        config = ReplacementConfig(team_count=12, roster_config=roster)
        with pytest.raises(FrozenInstanceError):
            config.team_count = 10  # type: ignore[misc]


class TestAssignPositions:
    def test_single_position_players_assigned_correctly(self) -> None:
        """Players eligible at only one position get assigned there."""
        players = [
            _make_player("p1", 10.0),
            _make_player("p2", 8.0),
            _make_player("p3", 6.0),
        ]
        positions: dict[str, tuple[str, ...]] = {
            "p1": ("C",),
            "p2": ("C",),
            "p3": ("C",),
        }
        roster = RosterConfig(slots=(RosterSlot("C", 1),))
        config = ReplacementConfig(team_count=2, roster_config=roster)

        result = assign_positions(
            players, positions, config, scarcity_order=("C",)
        )

        # 2 teams * 1 C slot = 2 rostered catchers
        assert result == {"p1": "C", "p2": "C"}
        # p3 is unrosterable, omitted
        assert "p3" not in result

    def test_multi_position_player_assigned_to_scarcest(self) -> None:
        """A C/1B player should go to C since C is scarcer."""
        players = [
            _make_player("p1", 10.0),  # C/1B eligible
            _make_player("p2", 8.0),  # 1B only
        ]
        positions: dict[str, tuple[str, ...]] = {
            "p1": ("C", "1B"),
            "p2": ("1B",),
        }
        roster = RosterConfig(
            slots=(RosterSlot("C", 1), RosterSlot("1B", 1))
        )
        config = ReplacementConfig(team_count=1, roster_config=roster)

        result = assign_positions(
            players, positions, config, scarcity_order=("C", "1B")
        )

        assert result["p1"] == "C"
        assert result["p2"] == "1B"

    def test_multi_position_overflow_falls_to_less_scarce(self) -> None:
        """When C is full, a C/1B player fills 1B instead."""
        players = [
            _make_player("p1", 10.0),  # C only
            _make_player("p2", 9.0),  # C/1B eligible
            _make_player("p3", 7.0),  # 1B only
        ]
        positions: dict[str, tuple[str, ...]] = {
            "p1": ("C",),
            "p2": ("C", "1B"),
            "p3": ("1B",),
        }
        roster = RosterConfig(
            slots=(RosterSlot("C", 1), RosterSlot("1B", 1))
        )
        config = ReplacementConfig(team_count=1, roster_config=roster)

        result = assign_positions(
            players, positions, config, scarcity_order=("C", "1B")
        )

        assert result["p1"] == "C"
        assert result["p2"] == "1B"
        assert "p3" not in result

    def test_remaining_batters_fill_util(self) -> None:
        """Players left after position assignment fill Util slots."""
        players = [
            _make_player("p1", 10.0),
            _make_player("p2", 8.0),
            _make_player("p3", 6.0),
        ]
        positions: dict[str, tuple[str, ...]] = {
            "p1": ("C",),
            "p2": ("C",),
            "p3": ("C",),
        }
        roster = RosterConfig(
            slots=(RosterSlot("C", 1), RosterSlot("Util", 1))
        )
        config = ReplacementConfig(team_count=1, roster_config=roster)

        result = assign_positions(
            players, positions, config, scarcity_order=("C",)
        )

        assert result["p1"] == "C"
        assert result["p2"] == "Util"
        assert "p3" not in result

    def test_util_respects_slot_count(self) -> None:
        """Only team_count * util_count players fill Util."""
        players = [_make_player(f"p{i}", 10.0 - i) for i in range(6)]
        positions: dict[str, tuple[str, ...]] = {
            f"p{i}": ("1B",) for i in range(6)
        }
        roster = RosterConfig(
            slots=(RosterSlot("1B", 1), RosterSlot("Util", 1))
        )
        config = ReplacementConfig(team_count=2, roster_config=roster)

        result = assign_positions(
            players, positions, config, scarcity_order=("1B",)
        )

        # 2 teams * 1 1B = 2 at 1B, 2 teams * 1 Util = 2 at Util
        assert sum(1 for v in result.values() if v == "1B") == 2
        assert sum(1 for v in result.values() if v == "Util") == 2
        assert len(result) == 4


class TestComputeReplacementLevels:
    def test_replacement_level_at_correct_rank(self) -> None:
        """With smoothing_window=1, replacement value is exactly the player at the threshold."""
        # 5 players, 1 team, 2 C slots → threshold at index 2 (0-indexed)
        players = [_make_player(f"p{i}", 10.0 - i) for i in range(5)]
        positions: dict[str, tuple[str, ...]] = {
            f"p{i}": ("C",) for i in range(5)
        }
        roster = RosterConfig(slots=(RosterSlot("C", 2),))
        config = ReplacementConfig(
            team_count=1, roster_config=roster, smoothing_window=1
        )

        thresholds = compute_replacement_levels(
            players, positions, config, scarcity_order=("C",)
        )

        assert len(thresholds) == 1
        t = thresholds[0]
        assert t.position == "C"
        assert t.roster_spots == 2
        assert t.replacement_rank == 2
        # Position pool: all 5 sorted desc. Assigned: p0, p1 at C.
        # Unassigned eligible: p2, p3, p4.
        # Pool = assigned + unassigned eligible = p0(10), p1(9), p2(8), p3(7), p4(6)
        # Threshold at index 2 → p2 with value 8.0
        assert t.replacement_value == 8.0

    def test_smoothing_window_averages_correctly(self) -> None:
        """Window=3 averages the player at threshold and one on each side."""
        # 7 players, 1 team, 2 C slots → threshold at index 2
        players = [_make_player(f"p{i}", 10.0 - i) for i in range(7)]
        positions: dict[str, tuple[str, ...]] = {
            f"p{i}": ("C",) for i in range(7)
        }
        roster = RosterConfig(slots=(RosterSlot("C", 2),))
        config = ReplacementConfig(
            team_count=1, roster_config=roster, smoothing_window=3
        )

        thresholds = compute_replacement_levels(
            players, positions, config, scarcity_order=("C",)
        )

        # Pool sorted: p0(10), p1(9), p2(8), p3(7), p4(6), p5(5), p6(4)
        # Threshold at index 2, window=3 → indices 1,2,3 → values 9,8,7 → avg 8.0
        assert thresholds[0].replacement_value == pytest.approx(8.0)

    def test_smoothing_window_clamps_at_edges(self) -> None:
        """Window extends beyond end of pool and is clamped."""
        # 4 players, 1 team, 3 C slots → threshold at index 3
        players = [_make_player(f"p{i}", 10.0 - i) for i in range(4)]
        positions: dict[str, tuple[str, ...]] = {
            f"p{i}": ("C",) for i in range(4)
        }
        roster = RosterConfig(slots=(RosterSlot("C", 3),))
        config = ReplacementConfig(
            team_count=1, roster_config=roster, smoothing_window=5
        )

        thresholds = compute_replacement_levels(
            players, positions, config, scarcity_order=("C",)
        )

        # Pool: p0(10), p1(9), p2(8), p3(7)
        # Threshold at index 3, window=5 → half=2, start=max(0,1)=1, end=min(4,6)=4
        # indices 1,2,3 → values 9,8,7 → avg 8.0
        assert thresholds[0].replacement_value == pytest.approx(8.0)

    def test_smoothing_window_clamps_at_start(self) -> None:
        """Window extends before start of pool and is clamped."""
        # 6 players, 1 team, 1 C slot → threshold at index 1
        players = [_make_player(f"p{i}", 10.0 - i) for i in range(6)]
        positions: dict[str, tuple[str, ...]] = {
            f"p{i}": ("C",) for i in range(6)
        }
        roster = RosterConfig(slots=(RosterSlot("C", 1),))
        config = ReplacementConfig(
            team_count=1, roster_config=roster, smoothing_window=5
        )

        thresholds = compute_replacement_levels(
            players, positions, config, scarcity_order=("C",)
        )

        # Pool: p0(10), p1(9), p2(8), p3(7), p4(6), p5(5)
        # Threshold at index 1, window=5 → half=2, start=max(0,-1)=0, end=min(6,4)=4
        # indices 0,1,2,3 → values 10,9,8,7 → avg 8.5
        assert thresholds[0].replacement_value == pytest.approx(8.5)

    def test_position_fewer_players_than_roster_spots(self) -> None:
        """When fewer eligible players exist than roster spots, use last player's value."""
        players = [_make_player("p0", 10.0), _make_player("p1", 8.0)]
        positions: dict[str, tuple[str, ...]] = {
            "p0": ("C",),
            "p1": ("C",),
        }
        roster = RosterConfig(slots=(RosterSlot("C", 2),))
        config = ReplacementConfig(
            team_count=2, roster_config=roster, smoothing_window=1
        )

        thresholds = compute_replacement_levels(
            players, positions, config, scarcity_order=("C",)
        )

        # 2 teams * 2 slots = 4 roster spots, but only 2 players exist
        # Threshold index 4 > pool size 2 → use last player's value (8.0)
        assert thresholds[0].replacement_value == 8.0

    def test_empty_player_pool(self) -> None:
        """When no players exist for a position, replacement value is 0.0."""
        players: list[PlayerValue] = []
        positions: dict[str, tuple[str, ...]] = {}
        roster = RosterConfig(slots=(RosterSlot("C", 2),))
        config = ReplacementConfig(
            team_count=1, roster_config=roster, smoothing_window=1
        )

        thresholds = compute_replacement_levels(
            players, positions, config, scarcity_order=("C",)
        )

        assert thresholds[0].replacement_value == 0.0

    def test_player_not_in_positions_dict_is_skipped(self) -> None:
        """Players without position data are ignored entirely."""
        players = [
            _make_player("p0", 10.0),
            _make_player("p1", 8.0),  # no positions entry
        ]
        positions: dict[str, tuple[str, ...]] = {
            "p0": ("C",),
        }
        roster = RosterConfig(slots=(RosterSlot("C", 1),))
        config = ReplacementConfig(
            team_count=1, roster_config=roster, smoothing_window=1
        )

        result = assign_positions(
            players, positions, config, scarcity_order=("C",)
        )

        assert "p0" in result
        assert "p1" not in result


class TestPitcherPath:
    def test_pitcher_scarcity_order(self) -> None:
        """SP/RP dual-eligible pitcher assigned to SP first (scarcer)."""
        players = [
            _make_player("sp1", 10.0, "pitcher"),
            _make_player("dual1", 9.0, "pitcher"),
            _make_player("rp1", 5.0, "pitcher"),
        ]
        positions: dict[str, tuple[str, ...]] = {
            "sp1": ("SP",),
            "dual1": ("SP", "RP"),
            "rp1": ("RP",),
        }
        roster = RosterConfig(
            slots=(RosterSlot("SP", 1), RosterSlot("RP", 1))
        )
        config = ReplacementConfig(team_count=1, roster_config=roster)

        result = assign_positions(
            players, positions, config, scarcity_order=PITCHER_SCARCITY_ORDER
        )

        assert result["sp1"] == "SP"
        assert result["dual1"] == "RP"
        assert "rp1" not in result


class TestIntegration:
    def test_full_batter_pipeline(self) -> None:
        """2-team mini league with multiple positions and Util."""
        # C, SS, 1B slots + 1 Util, 2 teams
        roster = RosterConfig(
            slots=(
                RosterSlot("C", 1),
                RosterSlot("SS", 1),
                RosterSlot("1B", 1),
                RosterSlot("Util", 1),
            )
        )
        config = ReplacementConfig(
            team_count=2, roster_config=roster, smoothing_window=1
        )

        # Create players: 6C, 6SS, 6 1B (enough for position pools after Util)
        players: list[PlayerValue] = []
        positions: dict[str, tuple[str, ...]] = {}

        for i in range(6):
            pid = f"c{i}"
            players.append(_make_player(pid, 20.0 - i))
            positions[pid] = ("C",)

        for i in range(6):
            pid = f"ss{i}"
            players.append(_make_player(pid, 18.0 - i))
            positions[pid] = ("SS",)

        for i in range(6):
            pid = f"1b{i}"
            players.append(_make_player(pid, 15.0 - i))
            positions[pid] = ("1B",)

        result = assign_positions(
            players, positions, config, scarcity_order=BATTER_SCARCITY_ORDER
        )

        # 2 C, 2 SS, 2 1B assigned to positions
        assert sum(1 for v in result.values() if v == "C") == 2
        assert sum(1 for v in result.values() if v == "SS") == 2
        assert sum(1 for v in result.values() if v == "1B") == 2
        # 2 Util slots filled from remaining (top unassigned: c2(18), c3(17))
        assert sum(1 for v in result.values() if v == "Util") == 2
        assert len(result) == 8

        # Replacement levels
        thresholds = compute_replacement_levels(
            players, positions, config, scarcity_order=BATTER_SCARCITY_ORDER
        )

        threshold_map = {t.position: t for t in thresholds}
        assert "C" in threshold_map
        assert "SS" in threshold_map
        assert "1B" in threshold_map
        assert "Util" in threshold_map

        # C pool: assigned(c0=20, c1=19) + unassigned eligible(c4=16, c5=15)
        # (c2, c3 assigned to Util → excluded)
        # Threshold at index 2 → c4(16) = 16.0
        assert threshold_map["C"].replacement_value == 16.0
        # SS pool: assigned(ss0=18, ss1=17) + unassigned(ss2=16, ss3=15, ss4=14, ss5=13)
        # Threshold at index 2 → ss2(16) = 16.0
        assert threshold_map["SS"].replacement_value == 16.0
        # 1B pool: assigned(1b0=15, 1b1=14) + unassigned(1b2=13, 1b3=12, 1b4=11, 1b5=10)
        # Threshold at index 2 → 1b2(13) = 13.0
        assert threshold_map["1B"].replacement_value == 13.0


class TestApplyReplacementAdjustment:
    def test_assigned_player_gets_threshold_subtracted(self) -> None:
        """An assigned player's total_value is reduced by the position threshold."""
        player = _make_player("p1", 10.0)
        assignments = {"p1": "C"}
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=4.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {"p1": ("C",)}

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        assert len(result) == 1
        assert result[0].total_value == pytest.approx(6.0)

    def test_replacement_level_player_lands_at_zero(self) -> None:
        """A player whose value equals the threshold ends up at 0.0 VORP."""
        player = _make_player("p1", 4.0)
        assignments = {"p1": "C"}
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=4.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {"p1": ("C",)}

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        assert result[0].total_value == pytest.approx(0.0)

    def test_scarce_position_gets_larger_boost(self) -> None:
        """C with low replacement outranks OF with high replacement after VORP."""
        catcher = _make_player("c1", 8.0)
        outfielder = _make_player("of1", 10.0)
        assignments = {"c1": "C", "of1": "OF"}
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=2.0,
            ),
            PositionThreshold(
                position="OF", roster_spots=60, replacement_rank=60,
                replacement_value=6.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {"c1": ("C",), "of1": ("OF",)}

        result = apply_replacement_adjustment(
            [catcher, outfielder], assignments, thresholds, positions
        )

        result_map = {p.player_id: p for p in result}
        # C: 8 - 2 = 6; OF: 10 - 6 = 4 → catcher ranks higher
        assert result_map["c1"].total_value == pytest.approx(6.0)
        assert result_map["of1"].total_value == pytest.approx(4.0)
        assert result_map["c1"].total_value > result_map["of1"].total_value

    def test_unassigned_player_gets_negative_value(self) -> None:
        """Unassigned below-replacement player gets negative VORP using min eligible threshold."""
        player = _make_player("p1", 2.0)
        assignments: dict[str, str] = {}  # not assigned
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=4.0,
            ),
            PositionThreshold(
                position="1B", roster_spots=12, replacement_rank=12,
                replacement_value=6.0,
            ),
        ]
        # Eligible at C and 1B; min threshold is C's 4.0
        positions: dict[str, tuple[str, ...]] = {"p1": ("C", "1B")}

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        # 2.0 - 4.0 = -2.0 (uses min threshold = most favorable)
        assert result[0].total_value == pytest.approx(-2.0)

    def test_player_without_positions_returned_unchanged(self) -> None:
        """A player with no position data gets replacement_value=0.0."""
        player = _make_player("p1", 5.0)
        assignments: dict[str, str] = {}
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=4.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {}  # no entry for p1

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        assert result[0].total_value == pytest.approx(5.0)


def _make_player_with_categories(
    player_id: str,
    total_value: float,
    category_values: tuple[CategoryValue, ...],
    position_type: str = "batter",
) -> PlayerValue:
    return PlayerValue(
        player_id=player_id,
        name=f"Player {player_id}",
        category_values=category_values,
        total_value=total_value,
        position_type=position_type,
    )


class TestProportionalScaling:
    def test_category_values_scaled_proportionally(self) -> None:
        """Each category value is scaled by adjusted_total / raw_total."""
        categories = (
            CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=3.0),
            CategoryValue(category=StatCategory.R, raw_stat=80.0, value=2.0),
        )
        player = _make_player_with_categories("p1", 5.0, categories)
        assignments = {"p1": "C"}
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=2.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {"p1": ("C",)}

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        adjusted = result[0]
        # total: 5 - 2 = 3; scale = 3/5 = 0.6
        assert adjusted.total_value == pytest.approx(3.0)
        assert adjusted.category_values[0].value == pytest.approx(1.8)  # 3.0 * 0.6
        assert adjusted.category_values[1].value == pytest.approx(1.2)  # 2.0 * 0.6
        # raw_stat preserved
        assert adjusted.category_values[0].raw_stat == 30.0
        assert adjusted.category_values[1].raw_stat == 80.0

    def test_category_ratios_preserved_after_adjustment(self) -> None:
        """The ratio between category values is the same before and after."""
        categories = (
            CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=4.0),
            CategoryValue(category=StatCategory.SB, raw_stat=20.0, value=1.0),
        )
        player = _make_player_with_categories("p1", 5.0, categories)
        assignments = {"p1": "C"}
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=1.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {"p1": ("C",)}

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        adjusted = result[0]
        original_ratio = 4.0 / 1.0
        adjusted_ratio = adjusted.category_values[0].value / adjusted.category_values[1].value
        assert adjusted_ratio == pytest.approx(original_ratio)

    def test_zero_raw_total_sets_categories_to_zero(self) -> None:
        """When raw total is 0.0, all category values become 0.0."""
        categories = (
            CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=0.0),
            CategoryValue(category=StatCategory.R, raw_stat=80.0, value=0.0),
        )
        player = _make_player_with_categories("p1", 0.0, categories)
        assignments = {"p1": "C"}
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=2.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {"p1": ("C",)}

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        adjusted = result[0]
        assert adjusted.total_value == pytest.approx(-2.0)
        assert adjusted.category_values[0].value == 0.0
        assert adjusted.category_values[1].value == 0.0
        # raw_stat preserved
        assert adjusted.category_values[0].raw_stat == 30.0

    def test_composite_only_adjusts_total_value_only(self) -> None:
        """When category_values is empty (ml-ridge), only total_value is adjusted."""
        player = _make_player("p1", 10.0)
        assignments = {"p1": "C"}
        thresholds = [
            PositionThreshold(
                position="C", roster_spots=12, replacement_rank=12,
                replacement_value=3.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {"p1": ("C",)}

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        assert result[0].total_value == pytest.approx(7.0)
        assert result[0].category_values == ()

    def test_player_metadata_preserved(self) -> None:
        """player_id, name, and position_type are preserved through adjustment."""
        player = PlayerValue(
            player_id="abc123",
            name="Mike Trout",
            category_values=(),
            total_value=10.0,
            position_type="batter",
        )
        assignments = {"abc123": "OF"}
        thresholds = [
            PositionThreshold(
                position="OF", roster_spots=60, replacement_rank=60,
                replacement_value=3.0,
            ),
        ]
        positions: dict[str, tuple[str, ...]] = {"abc123": ("OF",)}

        result = apply_replacement_adjustment(
            [player], assignments, thresholds, positions
        )

        assert result[0].player_id == "abc123"
        assert result[0].name == "Mike Trout"
        assert result[0].position_type == "batter"

    def test_empty_players_returns_empty(self) -> None:
        """Empty input list returns empty output."""
        result = apply_replacement_adjustment([], {}, [], {})
        assert result == []

    def test_empty_thresholds_returns_unchanged(self) -> None:
        """With no thresholds, all replacement values are 0.0."""
        player = _make_player("p1", 5.0)
        positions: dict[str, tuple[str, ...]] = {"p1": ("C",)}

        result = apply_replacement_adjustment(
            [player], {}, [], positions
        )

        assert result[0].total_value == pytest.approx(5.0)
