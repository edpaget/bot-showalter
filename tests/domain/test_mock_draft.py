from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.mock_draft import (
    BatchSimulationResult,
    BotStrategy,
    DraftPick,
    DraftResult,
    PlayerDraftFrequency,
    SimulationSummary,
    StrategyComparison,
)


class TestBotStrategy:
    def test_enum_values(self) -> None:
        assert BotStrategy.ADP_BASED == "adp_based"
        assert BotStrategy.BEST_VALUE == "best_value"
        assert BotStrategy.POSITIONAL_NEED == "positional_need"
        assert BotStrategy.RANDOM == "random"

    def test_enum_has_four_members(self) -> None:
        assert len(BotStrategy) == 4


class TestDraftPick:
    def test_frozen(self) -> None:
        pick = DraftPick(
            round=1,
            pick=1,
            team_idx=0,
            player_id=100,
            player_name="Test Player",
            position="C",
            value=25.0,
        )
        with pytest.raises(FrozenInstanceError):
            pick.round = 2  # type: ignore[misc]

    def test_fields(self) -> None:
        pick = DraftPick(
            round=3,
            pick=25,
            team_idx=0,
            player_id=42,
            player_name="Player X",
            position="OF",
            value=18.5,
        )
        assert pick.round == 3
        assert pick.pick == 25
        assert pick.team_idx == 0
        assert pick.player_id == 42
        assert pick.player_name == "Player X"
        assert pick.position == "OF"
        assert pick.value == 18.5


class TestDraftResult:
    def test_frozen(self) -> None:
        result = DraftResult(picks=[], rosters={}, snake=True)
        with pytest.raises(FrozenInstanceError):
            result.snake = False  # type: ignore[misc]

    def test_fields(self) -> None:
        pick = DraftPick(
            round=1,
            pick=1,
            team_idx=0,
            player_id=1,
            player_name="P",
            position="C",
            value=10.0,
        )
        result = DraftResult(picks=[pick], rosters={0: [pick]}, snake=True)
        assert len(result.picks) == 1
        assert result.rosters[0] == [pick]
        assert result.snake is True


class TestSimulationSummary:
    def test_frozen(self) -> None:
        s = SimulationSummary(
            n_simulations=10,
            team_idx=0,
            avg_roster_value=100.0,
            median_roster_value=99.0,
            p10_roster_value=80.0,
            p25_roster_value=90.0,
            p75_roster_value=110.0,
            p90_roster_value=120.0,
        )
        with pytest.raises(FrozenInstanceError):
            s.n_simulations = 5  # type: ignore[misc]

    def test_fields(self) -> None:
        s = SimulationSummary(
            n_simulations=50,
            team_idx=None,
            avg_roster_value=95.5,
            median_roster_value=94.0,
            p10_roster_value=80.0,
            p25_roster_value=88.0,
            p75_roster_value=105.0,
            p90_roster_value=110.0,
        )
        assert s.n_simulations == 50
        assert s.team_idx is None
        assert s.avg_roster_value == 95.5
        assert s.median_roster_value == 94.0
        assert s.p10_roster_value == 80.0
        assert s.p25_roster_value == 88.0
        assert s.p75_roster_value == 105.0
        assert s.p90_roster_value == 110.0


class TestPlayerDraftFrequency:
    def test_frozen(self) -> None:
        f = PlayerDraftFrequency(
            player_id=1,
            player_name="Test",
            pct_drafted=0.5,
            avg_round_drafted=3.0,
            avg_pick_drafted=25.0,
        )
        with pytest.raises(FrozenInstanceError):
            f.pct_drafted = 0.0  # type: ignore[misc]

    def test_fields(self) -> None:
        f = PlayerDraftFrequency(
            player_id=42,
            player_name="Player X",
            pct_drafted=0.75,
            avg_round_drafted=2.5,
            avg_pick_drafted=18.0,
        )
        assert f.player_id == 42
        assert f.player_name == "Player X"
        assert f.pct_drafted == 0.75
        assert f.avg_round_drafted == 2.5
        assert f.avg_pick_drafted == 18.0


class TestStrategyComparison:
    def test_frozen(self) -> None:
        sc = StrategyComparison(strategy_name="user", avg_value=100.0, win_rate=0.5)
        with pytest.raises(FrozenInstanceError):
            sc.win_rate = 0.0  # type: ignore[misc]

    def test_fields(self) -> None:
        sc = StrategyComparison(strategy_name="best_value", avg_value=120.0, win_rate=0.6)
        assert sc.strategy_name == "best_value"
        assert sc.avg_value == 120.0
        assert sc.win_rate == 0.6


class TestBatchSimulationResult:
    def test_frozen(self) -> None:
        summary = SimulationSummary(
            n_simulations=5,
            team_idx=0,
            avg_roster_value=100.0,
            median_roster_value=99.0,
            p10_roster_value=80.0,
            p25_roster_value=90.0,
            p75_roster_value=110.0,
            p90_roster_value=120.0,
        )
        result = BatchSimulationResult(
            summary=summary,
            player_frequencies=[],
            strategy_comparisons=[],
        )
        with pytest.raises(FrozenInstanceError):
            result.summary = summary  # type: ignore[misc]

    def test_fields(self) -> None:
        summary = SimulationSummary(
            n_simulations=10,
            team_idx=2,
            avg_roster_value=100.0,
            median_roster_value=99.0,
            p10_roster_value=80.0,
            p25_roster_value=90.0,
            p75_roster_value=110.0,
            p90_roster_value=120.0,
        )
        freq = PlayerDraftFrequency(
            player_id=1, player_name="P1", pct_drafted=0.3, avg_round_drafted=2.0, avg_pick_drafted=15.0
        )
        comp = StrategyComparison(strategy_name="user", avg_value=100.0, win_rate=0.5)
        result = BatchSimulationResult(
            summary=summary,
            player_frequencies=[freq],
            strategy_comparisons=[comp],
        )
        assert result.summary is summary
        assert len(result.player_frequencies) == 1
        assert len(result.strategy_comparisons) == 1
