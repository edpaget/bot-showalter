"""Tests for the mock draft simulation engine."""

import random

from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.mock_draft import (
    BatchSimulationResult,
    DraftPick,
    PlayerDraftFrequency,
    SimulationSummary,
    StrategyComparison,
)
from fantasy_baseball_manager.services.draft_state import build_draft_roster_slots
from fantasy_baseball_manager.services.mock_draft import _assign_position, run_batch_simulation, run_mock_draft
from fantasy_baseball_manager.services.mock_draft_bots import (
    ADPBot,
    BestValueBot,
    CategoryNeedRule,
    CompositeBot,
    FallbackBestValueRule,
    PositionalNeedBot,
    PositionTargetRule,
    RandomBot,
    TierValueRule,
    WeightedRule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BATTING_CAT = CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
_PITCHING_CAT = CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER)


def _make_row(
    player_id: int,
    name: str,
    position: str,
    value: float,
    *,
    adp: float | None = None,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type="B" if position not in ("SP", "RP") else "P",
        position=position,
        value=value,
        category_z_scores={},
        adp_overall=adp if adp is not None else float(player_id),
    )


def _small_league() -> LeagueSettings:
    """4-team league with minimal roster: C, 1B, OF, UTIL, P."""
    return LeagueSettings(
        name="Small Test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=4,
        budget=260,
        roster_batters=3,
        roster_pitchers=1,
        batting_categories=(_BATTING_CAT,),
        pitching_categories=(_PITCHING_CAT,),
        positions={"C": 1, "1B": 1, "OF": 1},
        roster_util=1,
    )


def _small_board() -> DraftBoard:
    """Board with plenty of players for 4 teams × 5 slots = 20 picks."""
    positions = ["C", "1B", "OF", "SP"]
    rows: list[DraftBoardRow] = []
    for i in range(40):
        pos = positions[i % len(positions)]
        rows.append(_make_row(i + 1, f"Player {i + 1}", pos, 40.0 - i, adp=float(i + 1)))
    return DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))


def _12_team_league() -> LeagueSettings:
    """Realistic 12-team league."""
    return LeagueSettings(
        name="Standard 12-Team",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=10,
        roster_pitchers=8,
        batting_categories=(_BATTING_CAT,),
        pitching_categories=(_PITCHING_CAT,),
        positions={"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "MI": 1, "CI": 1},
        roster_util=1,
    )


def _12_team_board() -> DraftBoard:
    """Board with enough players for 12 teams × 20 slots = 240 picks.

    Position cycle mirrors a realistic draft pool: ~40% pitchers, ~60% batters
    with good coverage at each position including composite-eligible ones.
    """
    rows: list[DraftBoardRow] = []
    # 10 batters + 8 pitchers per cycle = 18 positions
    positions = [
        "C",
        "1B",
        "2B",
        "3B",
        "SS",
        "OF",
        "OF",
        "OF",
        "1B",
        "3B",
        "SP",
        "SP",
        "SP",
        "SP",
        "RP",
        "RP",
        "RP",
        "RP",
    ]
    pid = 1
    for i in range(360):
        pos = positions[i % len(positions)]
        ptype = "B" if pos not in ("SP", "RP") else "P"
        rows.append(
            DraftBoardRow(
                player_id=pid,
                player_name=f"Player {pid}",
                rank=pid,
                player_type=ptype,
                position=pos,
                value=360.0 - i,
                category_z_scores={},
                adp_overall=float(pid),
            )
        )
        pid += 1
    return DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))


def _make_bots(n: int, seed: int = 42) -> list[ADPBot | BestValueBot | PositionalNeedBot | RandomBot]:
    """Create a mix of bot strategies for n teams."""
    rng = random.Random(seed)
    bot_classes = [ADPBot, BestValueBot, PositionalNeedBot, RandomBot]
    bots: list[ADPBot | BestValueBot | PositionalNeedBot | RandomBot] = []
    for i in range(n):
        cls = bot_classes[i % len(bot_classes)]
        bots.append(cls(rng=random.Random(rng.randint(0, 2**32))))
    return bots


# ---------------------------------------------------------------------------
# Snake ordering tests
# ---------------------------------------------------------------------------


class TestSnakeOrdering:
    def test_round_1_ascending(self) -> None:
        """Round 1: team 0, 1, 2, 3 (ascending)."""
        league = _small_league()
        board = _small_board()
        bots = [BestValueBot(rng=random.Random(i)) for i in range(4)]
        result = run_mock_draft(board, league, bots, snake=True, seed=42)

        # First 4 picks should be teams 0, 1, 2, 3
        round_1 = [p for p in result.picks if p.round == 1]
        assert [p.team_idx for p in round_1] == [0, 1, 2, 3]

    def test_round_2_descending(self) -> None:
        """Round 2: team 3, 2, 1, 0 (descending)."""
        league = _small_league()
        board = _small_board()
        bots = [BestValueBot(rng=random.Random(i)) for i in range(4)]
        result = run_mock_draft(board, league, bots, snake=True, seed=42)

        round_2 = [p for p in result.picks if p.round == 2]
        assert [p.team_idx for p in round_2] == [3, 2, 1, 0]

    def test_non_snake_always_ascending(self) -> None:
        """Non-snake: every round is 0, 1, 2, 3."""
        league = _small_league()
        board = _small_board()
        bots = [BestValueBot(rng=random.Random(i)) for i in range(4)]
        result = run_mock_draft(board, league, bots, snake=False, seed=42)

        for rnd in range(1, 5):
            round_picks = [p for p in result.picks if p.round == rnd]
            assert [p.team_idx for p in round_picks] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Positional enforcement tests
# ---------------------------------------------------------------------------


class TestPositionalEnforcement:
    def test_no_roster_exceeds_slots(self) -> None:
        """No team's roster exceeds slot counts defined by the league."""
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)
        result = run_mock_draft(board, league, bots, seed=42)

        slots = build_draft_roster_slots(league)
        for team_idx, roster in result.rosters.items():
            filled: dict[str, int] = {}
            for pick in roster:
                filled[pick.position] = filled.get(pick.position, 0) + 1
            for pos, count in filled.items():
                assert count <= slots[pos], f"Team {team_idx} has {count} players at {pos}, max is {slots[pos]}"

    def test_full_draft_fills_all_slots(self) -> None:
        """Each team should have exactly total_slots picks."""
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)
        result = run_mock_draft(board, league, bots, seed=42)

        slots = build_draft_roster_slots(league)
        total_slots = sum(slots.values())
        for team_idx, roster in result.rosters.items():
            assert len(roster) == total_slots, f"Team {team_idx} has {len(roster)} picks, expected {total_slots}"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        league = _small_league()
        board = _small_board()

        bots1 = _make_bots(4, seed=99)
        result1 = run_mock_draft(board, league, bots1, seed=42)

        bots2 = _make_bots(4, seed=99)
        result2 = run_mock_draft(board, league, bots2, seed=42)

        assert len(result1.picks) == len(result2.picks)
        for p1, p2 in zip(result1.picks, result2.picks, strict=True):
            assert p1 == p2


# ---------------------------------------------------------------------------
# 12-team integration test
# ---------------------------------------------------------------------------


class TestFullDraft:
    def test_12_team_mixed_strategies(self) -> None:
        """AC: run_mock_draft() completes a full draft for a 12-team league with mixed bot strategies."""
        league = _12_team_league()
        board = _12_team_board()
        bots = _make_bots(12, seed=42)
        result = run_mock_draft(board, league, bots, seed=123)

        slots = build_draft_roster_slots(league)
        total_slots = sum(slots.values())

        # All 12 teams filled
        assert len(result.rosters) == 12
        for team_idx in range(12):
            assert len(result.rosters[team_idx]) == total_slots

        # Total picks
        assert len(result.picks) == 12 * total_slots

        # Snake ordering check for round 1 and 2
        round_1 = [p for p in result.picks if p.round == 1]
        assert [p.team_idx for p in round_1] == list(range(12))
        round_2 = [p for p in result.picks if p.round == 2]
        assert [p.team_idx for p in round_2] == list(range(11, -1, -1))

        # Positional limits
        for team_idx, roster in result.rosters.items():
            filled: dict[str, int] = {}
            for pick in roster:
                filled[pick.position] = filled.get(pick.position, 0) + 1
            for pos, count in filled.items():
                assert count <= slots[pos], f"Team {team_idx} has {count} at {pos}, max {slots[pos]}"

    def test_12_team_pitcher_sub_slots(self) -> None:
        """Mock draft with SP/RP/P sub-slots completes without exceeding any slot."""
        league = LeagueSettings(
            name="Pitcher Sub-Slots 12-Team",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=12,
            budget=260,
            roster_batters=10,
            roster_pitchers=8,
            batting_categories=(_BATTING_CAT,),
            pitching_categories=(_PITCHING_CAT,),
            positions={"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "MI": 1, "CI": 1},
            roster_util=1,
            pitcher_positions={"SP": 2, "RP": 2, "P": 4},
        )
        board = _12_team_board()
        bots = _make_bots(12, seed=42)
        result = run_mock_draft(board, league, bots, seed=123)

        slots = build_draft_roster_slots(league)
        total_slots = sum(slots.values())

        # All 12 teams filled
        assert len(result.rosters) == 12
        for team_idx in range(12):
            assert len(result.rosters[team_idx]) == total_slots

        # Positional limits — SP, RP, P are separate slots
        assert "SP" in slots
        assert "RP" in slots
        assert "P" in slots
        for team_idx, roster in result.rosters.items():
            filled: dict[str, int] = {}
            for pick in roster:
                filled[pick.position] = filled.get(pick.position, 0) + 1
            for pos, count in filled.items():
                assert count <= slots[pos], f"Team {team_idx} has {count} at {pos}, max {slots[pos]}"

    def test_12_team_deterministic(self) -> None:
        """AC: Each bot strategy produces deterministic results given a fixed random seed."""
        league = _12_team_league()
        board = _12_team_board()

        bots1 = _make_bots(12, seed=42)
        result1 = run_mock_draft(board, league, bots1, seed=123)

        bots2 = _make_bots(12, seed=42)
        result2 = run_mock_draft(board, league, bots2, seed=123)

        for p1, p2 in zip(result1.picks, result2.picks, strict=True):
            assert p1 == p2


# ---------------------------------------------------------------------------
# DraftResult structure tests
# ---------------------------------------------------------------------------


class TestDraftResult:
    def test_snake_flag(self) -> None:
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)

        result_snake = run_mock_draft(board, league, bots, snake=True, seed=42)
        assert result_snake.snake is True

        bots2 = _make_bots(4)
        result_linear = run_mock_draft(board, league, bots2, snake=False, seed=42)
        assert result_linear.snake is False

    def test_picks_have_correct_round_numbers(self) -> None:
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)
        result = run_mock_draft(board, league, bots, seed=42)

        slots = build_draft_roster_slots(league)
        total_rounds = sum(slots.values())

        for pick in result.picks:
            assert 1 <= pick.round <= total_rounds
            assert 1 <= pick.pick <= len(result.picks)

    def test_no_duplicate_players(self) -> None:
        league = _small_league()
        board = _small_board()
        bots = _make_bots(4)
        result = run_mock_draft(board, league, bots, seed=42)

        player_ids = [p.player_id for p in result.picks]
        assert len(player_ids) == len(set(player_ids))


# ---------------------------------------------------------------------------
# Phase 2: CompositeBot acceptance criteria
# ---------------------------------------------------------------------------


def _ac_league() -> LeagueSettings:
    """4-team league with C, 1B, OF slots + UTIL for AC tests."""
    return LeagueSettings(
        name="AC Test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=4,
        budget=260,
        roster_batters=3,
        roster_pitchers=1,
        batting_categories=(_BATTING_CAT,),
        pitching_categories=(_PITCHING_CAT,),
        positions={"C": 1, "1B": 1, "OF": 1},
        roster_util=1,
    )


def _ac_board() -> DraftBoard:
    """Board for AC tests: catchers are low-valued so BestValue bots won't
    take them early, but PositionTargetRule can boost them."""
    rows: list[DraftBoardRow] = []
    # Non-catchers with high values
    non_c_positions = ["1B", "OF", "SP"]
    pid = 1
    for i in range(30):
        pos = non_c_positions[i % len(non_c_positions)]
        ptype = "B" if pos != "SP" else "P"
        rows.append(
            DraftBoardRow(
                player_id=pid,
                player_name=f"AC Player {pid}",
                rank=pid,
                player_type=ptype,
                position=pos,
                value=40.0 - i,
                category_z_scores={},
                adp_overall=float(pid),
            )
        )
        pid += 1
    # Catchers with distinctly lower values (5.0, 4.0, 3.0, ...)
    for i in range(10):
        rows.append(
            DraftBoardRow(
                player_id=pid,
                player_name=f"Catcher {pid}",
                rank=pid,
                player_type="B",
                position="C",
                value=5.0 - i * 0.5,
                category_z_scores={},
                adp_overall=float(pid),
            )
        )
        pid += 1
    return DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))


class TestCompositeBotPositionTarget:
    """AC1: CompositeBot with PositionTargetRule drafts a catcher in the target window."""

    def test_drafts_catcher_in_target_rounds(self) -> None:
        league = _ac_league()
        board = _ac_board()

        # Team 0 uses CompositeBot with PositionTargetRule for catchers in rounds 3-4
        composite = CompositeBot(
            rules=[
                WeightedRule(rule=PositionTargetRule(position="C", rounds=(3, 4)), weight=100.0),
                WeightedRule(rule=FallbackBestValueRule(), weight=1.0),
            ],
            rng=random.Random(42),
        )

        # Teams 1-3 are BestValue bots
        bots: list[CompositeBot | BestValueBot] = [composite]
        for i in range(3):
            bots.append(BestValueBot(rng=random.Random(i + 100)))

        result = run_mock_draft(board, league, bots, seed=42)

        # Team 0's roster should have a catcher drafted in round 3 or 4
        team0_roster = result.rosters[0]
        catcher_picks = [p for p in team0_roster if p.position == "C"]
        assert len(catcher_picks) >= 1
        catcher_rounds = {p.round for p in catcher_picks}
        assert catcher_rounds & {3, 4}, f"Expected catcher in round 3 or 4, got rounds {catcher_rounds}"


class TestCompositeBotCategoryNeed:
    """AC2: CategoryNeedRule shifts picks toward players improving the weakest category."""

    def test_category_need_picks_weak_category_player(self) -> None:
        """Build a board where CategoryNeedRule should prefer the SB-strong player."""
        # Players with distinct HR/SB z-scores
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="HR Star",
                rank=1,
                player_type="B",
                position="1B",
                value=20.0,
                category_z_scores={"HR": 3.0, "SB": 0.0},
                adp_overall=1.0,
            ),
            DraftBoardRow(
                player_id=2,
                player_name="SB Star",
                rank=2,
                player_type="B",
                position="OF",
                value=19.0,
                category_z_scores={"HR": 0.0, "SB": 3.0},
                adp_overall=2.0,
            ),
            DraftBoardRow(
                player_id=3,
                player_name="Balanced",
                rank=3,
                player_type="B",
                position="C",
                value=18.0,
                category_z_scores={"HR": 1.0, "SB": 1.0},
                adp_overall=3.0,
            ),
        ]

        # Roster already has an HR-heavy player (player 10)
        roster = [
            DraftPick(round=1, pick=1, team_idx=0, player_id=10, player_name="HR Guy", position="1B", value=25.0),
        ]
        z_lookup: dict[int, dict[str, float]] = {
            10: {"HR": 4.0, "SB": 0.0},
        }

        rule = CategoryNeedRule(z_score_lookup=z_lookup)

        # Score each player — SB Star should score highest (weakest cat is SB, and SB Star has z=3.0 in SB)
        league = _ac_league()
        scores = {row.player_name: rule.score(row, roster, 2, league) for row in rows}
        assert scores["SB Star"] == 3.0
        assert scores["HR Star"] is None  # z=0.0 in SB → not positive
        assert scores["Balanced"] == 1.0  # z=1.0 in SB

        # CompositeBot with CategoryNeedRule picks SB Star
        bot = CompositeBot(
            rules=[
                WeightedRule(rule=CategoryNeedRule(z_score_lookup=z_lookup), weight=10.0),
                WeightedRule(rule=FallbackBestValueRule(), weight=1.0),
            ],
            rng=random.Random(42),
        )
        pick_id = bot.pick(rows, roster, league)
        assert pick_id == 2  # SB Star


class TestCompositeBotComposability:
    """AC3: Rules compose cleanly — adding/removing rules changes behavior."""

    def test_adding_removing_rules_changes_pick(self) -> None:
        available = [
            DraftBoardRow(
                player_id=1,
                player_name="1B Target",
                rank=1,
                player_type="B",
                position="1B",
                value=20.0,
                category_z_scores={},
                adp_overall=1.0,
                tier=None,
            ),
            DraftBoardRow(
                player_id=2,
                player_name="Tier 1 C",
                rank=2,
                player_type="B",
                position="C",
                value=25.0,
                category_z_scores={},
                adp_overall=2.0,
                tier=1,
            ),
            DraftBoardRow(
                player_id=3,
                player_name="Good OF",
                rank=3,
                player_type="B",
                position="OF",
                value=22.0,
                category_z_scores={},
                adp_overall=3.0,
                tier=None,
            ),
        ]
        league = _ac_league()

        rule_a = WeightedRule(rule=FallbackBestValueRule(), weight=1.0)
        rule_b = WeightedRule(rule=TierValueRule(), weight=10.0)
        rule_c = WeightedRule(rule=PositionTargetRule(position="1B", rounds=(1, 5)), weight=5.0)

        # A+B+C: Tier1 C wins due to tier boost
        # id=1(1B): 1*20 + 10*None + 5*20 = 120
        # id=2(C,t1): 1*25 + 10*(25*2) + 5*None = 525
        # id=3(OF): 1*22 + 10*None + 5*None = 22
        bot_abc = CompositeBot(rules=[rule_a, rule_b, rule_c], rng=random.Random(42))
        pick_abc = bot_abc.pick(available, [], league)
        assert pick_abc == 2  # Tier 1 C wins

        # Remove rule B (TierValueRule) → 1B Target wins
        # id=1(1B): 1*20 + 5*20 = 120
        # id=2(C): 1*25 = 25 (no tier, no position target)
        # id=3(OF): 1*22 = 22
        bot_ac = CompositeBot(rules=[rule_a, rule_c], rng=random.Random(42))
        pick_ac = bot_ac.pick(available, [], league)
        assert pick_ac == 1  # 1B Target wins
        assert pick_ac != pick_abc  # Different pick

        # Restore rule B → same as original
        bot_abc_again = CompositeBot(rules=[rule_a, rule_b, rule_c], rng=random.Random(42))
        pick_abc_again = bot_abc_again.pick(available, [], league)
        assert pick_abc_again == pick_abc  # Same pick restored


# ---------------------------------------------------------------------------
# Phase 3: Batch simulation tests
# ---------------------------------------------------------------------------


class TestBatchSimulation:
    def test_returns_correct_types(self) -> None:
        """Small board, n=5, verify types and n_simulations."""
        board = _small_board()
        league = _small_league()
        result = run_batch_simulation(
            n_simulations=5,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: ADPBot(rng=rng) for _ in range(3)],
            seed=42,
        )
        assert isinstance(result, BatchSimulationResult)
        assert isinstance(result.summary, SimulationSummary)
        assert result.summary.n_simulations == 5
        assert all(isinstance(f, PlayerDraftFrequency) for f in result.player_frequencies)
        assert all(isinstance(c, StrategyComparison) for c in result.strategy_comparisons)

    def test_deterministic(self) -> None:
        """Same seed → identical results."""
        board = _small_board()
        league = _small_league()

        def _run() -> BatchSimulationResult:
            return run_batch_simulation(
                n_simulations=5,
                board=board,
                league=league,
                user_strategy_factory=lambda rng: BestValueBot(rng=rng),
                opponent_strategy_factories=[lambda rng: ADPBot(rng=rng) for _ in range(3)],
                seed=99,
            )

        r1 = _run()
        r2 = _run()
        assert r1.summary == r2.summary
        assert r1.player_frequencies == r2.player_frequencies
        assert r1.strategy_comparisons == r2.strategy_comparisons

    def test_pct_drafted_sums_to_roster_size(self) -> None:
        """sum(pf.pct_drafted) ≈ roster_size."""
        board = _small_board()
        league = _small_league()
        slots = build_draft_roster_slots(league)
        roster_size = sum(slots.values())
        result = run_batch_simulation(
            n_simulations=10,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: ADPBot(rng=rng) for _ in range(3)],
            seed=42,
        )
        total_pct = sum(f.pct_drafted for f in result.player_frequencies)
        assert abs(total_pct - roster_size) < 0.01

    def test_win_rate_sums_to_one(self) -> None:
        """sum(sc.win_rate) ≈ 1.0."""
        board = _small_board()
        league = _small_league()
        result = run_batch_simulation(
            n_simulations=10,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: RandomBot(rng=rng) for _ in range(3)],
            seed=42,
        )
        total_win_rate = sum(sc.win_rate for sc in result.strategy_comparisons)
        assert abs(total_win_rate - 1.0) < 0.01

    def test_identifies_best_strategy(self) -> None:
        """BestValueBot user vs RandomBot opponents → user has highest avg_value."""
        board = _small_board()
        league = _small_league()
        result = run_batch_simulation(
            n_simulations=20,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: RandomBot(rng=rng) for _ in range(3)],
            seed=42,
        )
        user_comp = next(sc for sc in result.strategy_comparisons if sc.strategy_name == "user")
        opponent_comps = [sc for sc in result.strategy_comparisons if sc.strategy_name != "user"]
        assert all(user_comp.avg_value >= oc.avg_value for oc in opponent_comps)

    def test_random_draft_position(self) -> None:
        """draft_position=None → summary.team_idx is None."""
        board = _small_board()
        league = _small_league()
        result = run_batch_simulation(
            n_simulations=5,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: ADPBot(rng=rng) for _ in range(3)],
            draft_position=None,
            seed=42,
        )
        assert result.summary.team_idx is None

    def test_fixed_draft_position(self) -> None:
        """draft_position=2 → summary.team_idx == 2."""
        board = _small_board()
        league = _small_league()
        result = run_batch_simulation(
            n_simulations=5,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: ADPBot(rng=rng) for _ in range(3)],
            draft_position=2,
            seed=42,
        )
        assert result.summary.team_idx == 2

    def test_percentile_ordering(self) -> None:
        """p10 <= p25 <= median <= p75 <= p90."""
        board = _small_board()
        league = _small_league()
        result = run_batch_simulation(
            n_simulations=20,
            board=board,
            league=league,
            user_strategy_factory=lambda rng: BestValueBot(rng=rng),
            opponent_strategy_factories=[lambda rng: ADPBot(rng=rng) for _ in range(3)],
            seed=42,
        )
        s = result.summary
        assert s.p10_roster_value <= s.p25_roster_value
        assert s.p25_roster_value <= s.median_roster_value
        assert s.median_roster_value <= s.p75_roster_value
        assert s.p75_roster_value <= s.p90_roster_value


# ---------------------------------------------------------------------------
# _assign_position tests (including bench fallback)
# ---------------------------------------------------------------------------


class TestAssignPosition:
    def test_primary_position(self) -> None:
        player = _make_row(1, "Batter", "1B", 10.0)
        assert _assign_position(player, {"1B": 1, "BN": 2}) == "1B"

    def test_composite_slot_mi(self) -> None:
        player = _make_row(1, "SS Guy", "SS", 10.0)
        assert _assign_position(player, {"MI": 1, "BN": 2}) == "MI"

    def test_composite_slot_ci(self) -> None:
        player = _make_row(1, "3B Guy", "3B", 10.0)
        assert _assign_position(player, {"CI": 1, "BN": 2}) == "CI"

    def test_util_for_batter(self) -> None:
        player = _make_row(1, "OF Guy", "OF", 10.0)
        assert _assign_position(player, {"UTIL": 1, "BN": 2}) == "UTIL"

    def test_flex_p_for_pitcher(self) -> None:
        player = _make_row(1, "Pitcher", "SP", 10.0)
        assert _assign_position(player, {"P": 1, "BN": 2}) == "P"

    def test_bench_fallback_for_batter(self) -> None:
        """Batter with no primary/composite/UTIL slots falls back to BN."""
        player = _make_row(1, "OF Guy", "OF", 10.0)
        assert _assign_position(player, {"BN": 3}) == "BN"

    def test_bench_fallback_for_pitcher(self) -> None:
        """Pitcher with no primary/P slots falls back to BN."""
        player = _make_row(1, "Pitcher", "SP", 10.0)
        assert _assign_position(player, {"BN": 3}) == "BN"

    def test_bench_fallback_only_when_needed(self) -> None:
        """BN is not used when a better slot is available."""
        player = _make_row(1, "1B Guy", "1B", 10.0)
        result = _assign_position(player, {"1B": 1, "BN": 5})
        assert result == "1B"

    def test_none_when_no_slots(self) -> None:
        player = _make_row(1, "OF Guy", "OF", 10.0)
        assert _assign_position(player, {}) is None

    def test_none_when_bench_full(self) -> None:
        player = _make_row(1, "OF Guy", "OF", 10.0)
        assert _assign_position(player, {"BN": 0}) is None


class TestMockDraftWithBench:
    """Integration test: mock draft with bench slots completes without error."""

    def test_league_with_bench_slots_completes(self) -> None:
        league = LeagueSettings(
            name="Bench Test",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=4,
            budget=260,
            roster_batters=2,
            roster_pitchers=1,
            roster_bench=3,
            batting_categories=(_BATTING_CAT,),
            pitching_categories=(_PITCHING_CAT,),
            positions={"C": 1, "OF": 1},
            roster_util=0,
        )
        board = _small_board()
        bots = [BestValueBot(rng=random.Random(i)) for i in range(4)]
        result = run_mock_draft(board, league, bots, seed=42)

        slots = build_draft_roster_slots(league)
        total_slots = sum(slots.values())
        assert "BN" in slots
        assert slots["BN"] == 3

        for team_idx in range(4):
            assert len(result.rosters[team_idx]) == total_slots

        # Verify bench slots are actually used
        all_positions = [p.position for p in result.picks]
        assert "BN" in all_positions


class TestPoolExhaustion:
    """Draft completes gracefully when the player pool is smaller than total roster slots."""

    def test_draft_ends_early_when_pool_exhausted(self) -> None:
        league = LeagueSettings(
            name="Exhaustion Test",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=2,
            budget=260,
            roster_batters=2,
            roster_pitchers=1,
            roster_bench=2,
            batting_categories=(_BATTING_CAT,),
            pitching_categories=(_PITCHING_CAT,),
            positions={"OF": 2},
            roster_util=0,
        )
        # 2 teams × 5 slots = 10 picks needed, but only 7 players
        rows = [
            _make_row(1, "OF1", "OF", 10.0),
            _make_row(2, "OF2", "OF", 9.0),
            _make_row(3, "OF3", "OF", 8.0),
            _make_row(4, "OF4", "OF", 7.0),
            _make_row(5, "SP1", "SP", 6.0),
            _make_row(6, "SP2", "SP", 5.0),
            _make_row(7, "OF5", "OF", 4.0),
        ]
        board = DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))
        bots = [BestValueBot(rng=random.Random(i)) for i in range(2)]

        result = run_mock_draft(board, league, bots, seed=42)

        assert len(result.picks) < 10
        assert len(result.picks) == 7
        # No duplicate players
        player_ids = [p.player_id for p in result.picks]
        assert len(player_ids) == len(set(player_ids))

    def test_draft_skips_team_with_no_assignable_players(self) -> None:
        """When one team has no positional needs that match remaining players, it's skipped."""
        league = LeagueSettings(
            name="Skip Test",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=2,
            budget=260,
            roster_batters=1,
            roster_pitchers=1,
            batting_categories=(_BATTING_CAT,),
            pitching_categories=(_PITCHING_CAT,),
            positions={"C": 1},
            roster_util=0,
        )
        # 2 teams × 2 slots = 4 picks, but only 1 catcher and 2 pitchers
        rows = [
            _make_row(1, "C1", "C", 10.0),
            _make_row(2, "SP1", "SP", 9.0),
            _make_row(3, "SP2", "SP", 8.0),
        ]
        board = DraftBoard(rows=rows, batting_categories=("HR",), pitching_categories=("K",))
        bots = [BestValueBot(rng=random.Random(i)) for i in range(2)]

        result = run_mock_draft(board, league, bots, seed=42)

        # Team 0 gets C1 + SP1, team 1 gets only SP2 (no catcher available)
        assert len(result.picks) == 3
        player_ids = [p.player_id for p in result.picks]
        assert len(player_ids) == len(set(player_ids))
