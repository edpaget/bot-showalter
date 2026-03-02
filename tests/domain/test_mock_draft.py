from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.mock_draft import (
    BotStrategy,
    DraftPick,
    DraftResult,
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
