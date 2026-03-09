"""Tests for draft plan generation from mock draft simulation results."""

from __future__ import annotations

import statistics

from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.domain.draft_plan import DraftPlan, DraftPlanTarget
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.mock_draft import DraftPick
from fantasy_baseball_manager.services.draft_plan import generate_draft_plan
from fantasy_baseball_manager.services.mock_draft import run_batch_simulation
from fantasy_baseball_manager.services.mock_draft_bots import ADPBot, BestValueBot


def _pick(
    rnd: int,
    position: str,
    player_name: str,
    *,
    pick: int = 1,
    team_idx: int = 0,
    player_id: int = 1,
    value: float = 10.0,
) -> DraftPick:
    return DraftPick(
        round=rnd,
        pick=pick,
        team_idx=team_idx,
        player_id=player_id,
        player_name=player_name,
        position=position,
        value=value,
    )


class TestBasicExtraction:
    """3 simulations where round 1 = OF, rounds 2-3 = SP → targets [(1,1,OF), (2,3,SP)]."""

    def test_extracts_round_position_targets(self) -> None:
        rosters = [
            [
                _pick(1, "OF", "Player A", player_id=1, value=20.0),
                _pick(2, "SP", "Player B", player_id=2, value=15.0),
                _pick(3, "SP", "Player C", player_id=3, value=10.0),
            ],
            [
                _pick(1, "OF", "Player D", player_id=4, value=19.0),
                _pick(2, "SP", "Player E", player_id=5, value=14.0),
                _pick(3, "SP", "Player F", player_id=6, value=9.0),
            ],
            [
                _pick(1, "OF", "Player G", player_id=7, value=18.0),
                _pick(2, "SP", "Player H", player_id=8, value=13.0),
                _pick(3, "SP", "Player I", player_id=9, value=8.0),
            ],
        ]

        plan = generate_draft_plan(rosters, slot=0, teams=12, strategy_name="test")

        assert isinstance(plan, DraftPlan)
        assert len(plan.targets) == 2
        assert plan.targets[0].round_range == (1, 1)
        assert plan.targets[0].position == "OF"
        assert plan.targets[1].round_range == (2, 3)
        assert plan.targets[1].position == "SP"
        assert plan.n_simulations == 3
        assert plan.slot == 0
        assert plan.teams == 12
        assert plan.strategy_name == "test"

    def test_targets_are_ordered_by_round(self) -> None:
        rosters = [
            [
                _pick(1, "OF", "A", player_id=1),
                _pick(2, "SP", "B", player_id=2),
                _pick(3, "C", "C", player_id=3),
            ],
        ]
        plan = generate_draft_plan(rosters, slot=0, teams=12, strategy_name="test")
        rounds = [t.round_range[0] for t in plan.targets]
        assert rounds == sorted(rounds)


class TestConfidenceCalculation:
    """If 8/10 sims draft SP in round 2, confidence = 0.8."""

    def test_confidence_reflects_position_frequency(self) -> None:
        rosters: list[list[DraftPick]] = []
        for i in range(8):
            rosters.append([_pick(2, "SP", f"SP Player {i}", player_id=i, value=10.0)])
        for i in range(2):
            rosters.append([_pick(2, "OF", f"OF Player {i}", player_id=100 + i, value=10.0)])

        plan = generate_draft_plan(rosters, slot=0, teams=12, strategy_name="test")

        assert len(plan.targets) == 1
        assert plan.targets[0].position == "SP"
        assert plan.targets[0].confidence == 0.8


class TestAdjacentRoundMerging:
    """Rounds 2,3,4 all SP → single DraftPlanTarget(round_range=(2,4), position='SP')."""

    def test_merges_adjacent_same_position(self) -> None:
        rosters = [
            [
                _pick(1, "OF", "A", player_id=1),
                _pick(2, "SP", "B", player_id=2),
                _pick(3, "SP", "C", player_id=3),
                _pick(4, "SP", "D", player_id=4),
                _pick(5, "2B", "E", player_id=5),
            ],
        ]

        plan = generate_draft_plan(rosters, slot=0, teams=12, strategy_name="test")

        assert len(plan.targets) == 3
        assert plan.targets[0] == DraftPlanTarget(
            round_range=(1, 1), position="OF", confidence=1.0, example_players=["A"]
        )
        assert plan.targets[1] == DraftPlanTarget(
            round_range=(2, 4), position="SP", confidence=1.0, example_players=["B", "C", "D"]
        )
        assert plan.targets[2] == DraftPlanTarget(
            round_range=(5, 5), position="2B", confidence=1.0, example_players=["E"]
        )

    def test_does_not_merge_non_adjacent_same_position(self) -> None:
        rosters = [
            [
                _pick(1, "SP", "A", player_id=1),
                _pick(2, "OF", "B", player_id=2),
                _pick(3, "SP", "C", player_id=3),
            ],
        ]

        plan = generate_draft_plan(rosters, slot=0, teams=12, strategy_name="test")

        assert len(plan.targets) == 3
        assert plan.targets[0].position == "SP"
        assert plan.targets[0].round_range == (1, 1)
        assert plan.targets[1].position == "OF"
        assert plan.targets[2].position == "SP"
        assert plan.targets[2].round_range == (3, 3)


class TestExamplePlayers:
    """Most frequent names at position-round appear in example_players, capped at 3."""

    def test_top_3_most_frequent_names(self) -> None:
        rosters: list[list[DraftPick]] = []
        # Alice appears 5 times, Bob 4, Charlie 3, Dave 2
        names_and_counts = [("Alice", 5), ("Bob", 4), ("Charlie", 3), ("Dave", 2)]
        pid = 1
        for name, count in names_and_counts:
            for _ in range(count):
                rosters.append([_pick(1, "SP", name, player_id=pid, value=10.0)])
                pid += 1

        plan = generate_draft_plan(rosters, slot=0, teams=12, strategy_name="test")

        assert plan.targets[0].example_players == ["Alice", "Bob", "Charlie"]

    def test_fewer_than_3_players(self) -> None:
        rosters = [
            [_pick(1, "SP", "Only One", player_id=1, value=10.0)],
        ]

        plan = generate_draft_plan(rosters, slot=0, teams=12, strategy_name="test")

        assert plan.targets[0].example_players == ["Only One"]


class TestWinningVsLosingSims:
    """Filtering to above-median rosters should yield higher confidence."""

    def test_winning_sims_have_higher_confidence(self) -> None:
        # Create rosters where high-value rosters consistently draft SP in round 2,
        # while low-value rosters are split between SP and OF
        rosters: list[list[DraftPick]] = []
        values: list[float] = []

        # 10 "winning" sims: all draft SP in round 2 (high value)
        for i in range(10):
            roster = [
                _pick(1, "OF", f"OF{i}", player_id=i * 10, value=30.0),
                _pick(2, "SP", f"SP{i}", player_id=i * 10 + 1, value=25.0),
            ]
            rosters.append(roster)
            values.append(sum(p.value for p in roster))

        # 10 "losing" sims: 5 draft SP, 5 draft OF in round 2 (low value)
        for i in range(5):
            roster = [
                _pick(1, "OF", f"LOF{i}", player_id=200 + i * 10, value=10.0),
                _pick(2, "SP", f"LSP{i}", player_id=200 + i * 10 + 1, value=5.0),
            ]
            rosters.append(roster)
            values.append(sum(p.value for p in roster))
        for i in range(5):
            roster = [
                _pick(1, "OF", f"LOF2{i}", player_id=300 + i * 10, value=10.0),
                _pick(2, "OF", f"LOF3{i}", player_id=300 + i * 10 + 1, value=5.0),
            ]
            rosters.append(roster)
            values.append(sum(p.value for p in roster))

        median_val = statistics.median(values)

        winning_rosters = [r for r, v in zip(rosters, values, strict=True) if v >= median_val]
        losing_rosters = [r for r, v in zip(rosters, values, strict=True) if v < median_val]

        winning_plan = generate_draft_plan(winning_rosters, slot=0, teams=12, strategy_name="win")
        losing_plan = generate_draft_plan(losing_rosters, slot=0, teams=12, strategy_name="lose")

        # Round 2 target for winning sims
        winning_r2 = next(t for t in winning_plan.targets if t.round_range[0] <= 2 <= t.round_range[1])
        losing_r2 = next(t for t in losing_plan.targets if t.round_range[0] <= 2 <= t.round_range[1])

        assert winning_r2.position == "SP"
        assert winning_r2.confidence > losing_r2.confidence


class TestEdgeCases:
    def test_single_simulation(self) -> None:
        rosters = [
            [
                _pick(1, "OF", "A", player_id=1, value=20.0),
                _pick(2, "SP", "B", player_id=2, value=15.0),
            ],
        ]

        plan = generate_draft_plan(rosters, slot=3, teams=10, strategy_name="solo")

        assert plan.n_simulations == 1
        assert len(plan.targets) == 2
        assert plan.targets[0].confidence == 1.0
        assert plan.targets[1].confidence == 1.0
        assert plan.avg_roster_value == 35.0

    def test_single_round(self) -> None:
        rosters = [
            [_pick(1, "OF", "A", player_id=1, value=20.0)],
            [_pick(1, "OF", "B", player_id=2, value=18.0)],
        ]

        plan = generate_draft_plan(rosters, slot=0, teams=12, strategy_name="test")

        assert len(plan.targets) == 1
        assert plan.targets[0].round_range == (1, 1)
        assert plan.targets[0].position == "OF"
        assert plan.targets[0].confidence == 1.0

    def test_empty_rosters(self) -> None:
        plan = generate_draft_plan([], slot=0, teams=12, strategy_name="empty")

        assert plan.n_simulations == 0
        assert plan.targets == []
        assert plan.avg_roster_value == 0.0


class TestBatchSimulationUserRosters:
    """Verify BatchSimulationResult includes user_rosters after step 2 changes."""

    def test_user_rosters_length_equals_n_simulations(self) -> None:
        cat = CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
        pcat = CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER)

        league = LeagueSettings(
            name="Test",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=4,
            budget=260,
            roster_batters=3,
            roster_pitchers=1,
            batting_categories=(cat,),
            pitching_categories=(pcat,),
            positions={"C": 1, "1B": 1, "OF": 1},
            roster_util=1,
        )

        positions = ["C", "1B", "OF", "SP"]
        rows = [
            DraftBoardRow(
                player_id=i + 1,
                player_name=f"P{i + 1}",
                rank=i + 1,
                player_type="B" if positions[i % 4] != "SP" else "P",
                position=positions[i % 4],
                value=40.0 - i,
                category_z_scores={},
                adp_overall=float(i + 1),
            )
            for i in range(40)
        ]
        board = DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))

        n_sims = 5
        result = run_batch_simulation(
            n_simulations=n_sims,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: ADPBot(rng=rng) for _ in range(3)],
            seed=42,
        )

        assert len(result.user_rosters) == n_sims
        assert len(result.user_roster_values) == n_sims
        for roster in result.user_rosters:
            assert len(roster) > 0
        for val in result.user_roster_values:
            assert val > 0.0

    def test_all_player_picks_populated(self) -> None:
        cat = CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
        pcat = CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER)

        league = LeagueSettings(
            name="Test",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=4,
            budget=260,
            roster_batters=3,
            roster_pitchers=1,
            batting_categories=(cat,),
            pitching_categories=(pcat,),
            positions={"C": 1, "1B": 1, "OF": 1},
            roster_util=1,
        )

        positions = ["C", "1B", "OF", "SP"]
        rows = [
            DraftBoardRow(
                player_id=i + 1,
                player_name=f"P{i + 1}",
                rank=i + 1,
                player_type="B" if positions[i % 4] != "SP" else "P",
                position=positions[i % 4],
                value=40.0 - i,
                category_z_scores={},
                adp_overall=float(i + 1),
            )
            for i in range(40)
        ]
        board = DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))

        n_sims = 5
        result = run_batch_simulation(
            n_simulations=n_sims,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: ADPBot(rng=rng) for _ in range(3)],
            seed=42,
        )

        # all_player_picks should contain entries for drafted players
        assert len(result.all_player_picks) > 0
        for _player_id, pick_numbers in result.all_player_picks.items():
            assert len(pick_numbers) > 0
            # Each pick number should be a positive integer
            for pick in pick_numbers:
                assert pick > 0
            # Should have at most n_sims entries per player
            assert len(pick_numbers) <= n_sims
