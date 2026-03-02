from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.mock_draft import DraftPick
from fantasy_baseball_manager.domain.pick_value import (
    CascadeResult,
    CascadeRoster,
    PickTrade,
    PickTradeEvaluation,
    PickValue,
    PickValueCurve,
)
from fantasy_baseball_manager.services.pick_value import value_at


class TestPickValue:
    def test_frozen(self) -> None:
        pv = PickValue(pick=1, expected_value=30.0, player_name="Player A", confidence="high")
        with pytest.raises(FrozenInstanceError):
            pv.pick = 2  # type: ignore[misc]

    def test_fields(self) -> None:
        pv = PickValue(pick=5, expected_value=20.0, player_name="Player B", confidence="medium")
        assert pv.pick == 5
        assert pv.expected_value == 20.0
        assert pv.player_name == "Player B"
        assert pv.confidence == "medium"

    def test_player_name_optional(self) -> None:
        pv = PickValue(pick=10, expected_value=5.0, player_name=None, confidence="low")
        assert pv.player_name is None


class TestPickValueCurve:
    def _curve(self, picks: list[PickValue] | None = None) -> PickValueCurve:
        if picks is None:
            picks = [
                PickValue(pick=1, expected_value=30.0, player_name="A", confidence="high"),
                PickValue(pick=2, expected_value=25.0, player_name="B", confidence="high"),
                PickValue(pick=3, expected_value=20.0, player_name="C", confidence="medium"),
            ]
        return PickValueCurve(
            season=2026,
            provider="yahoo",
            system="zar",
            picks=picks,
            total_picks=3,
        )

    def test_frozen(self) -> None:
        curve = self._curve()
        with pytest.raises(FrozenInstanceError):
            curve.season = 2025  # type: ignore[misc]

    def test_fields(self) -> None:
        curve = self._curve()
        assert curve.season == 2026
        assert curve.provider == "yahoo"
        assert curve.system == "zar"
        assert curve.total_picks == 3
        assert len(curve.picks) == 3

    def test_value_at_returns_expected(self) -> None:
        curve = self._curve()
        assert value_at(curve, 1) == 30.0
        assert value_at(curve, 2) == 25.0
        assert value_at(curve, 3) == 20.0

    def test_value_at_out_of_range_returns_zero(self) -> None:
        curve = self._curve()
        assert value_at(curve, 0) == 0.0
        assert value_at(curve, 4) == 0.0
        assert value_at(curve, -1) == 0.0
        assert value_at(curve, 999) == 0.0

    def test_value_at_empty_curve(self) -> None:
        curve = self._curve(picks=[])
        assert value_at(curve, 1) == 0.0


class TestPickTrade:
    def test_frozen(self) -> None:
        trade = PickTrade(gives=[1], receives=[10])
        with pytest.raises(FrozenInstanceError):
            trade.gives = [2]  # type: ignore[misc]

    def test_fields(self) -> None:
        trade = PickTrade(gives=[1, 20], receives=[10, 15])
        assert trade.gives == [1, 20]
        assert trade.receives == [10, 15]


class TestPickTradeEvaluation:
    def _pv(self, pick: int, value: float) -> PickValue:
        return PickValue(pick=pick, expected_value=value, player_name=None, confidence="high")

    def test_frozen(self) -> None:
        trade = PickTrade(gives=[1], receives=[10])
        evaluation = PickTradeEvaluation(
            trade=trade,
            gives_value=30.0,
            receives_value=20.0,
            net_value=-10.0,
            gives_detail=[self._pv(1, 30.0)],
            receives_detail=[self._pv(10, 20.0)],
            recommendation="reject",
        )
        with pytest.raises(FrozenInstanceError):
            evaluation.net_value = 0.0  # type: ignore[misc]

    def test_fields(self) -> None:
        trade = PickTrade(gives=[1], receives=[10])
        evaluation = PickTradeEvaluation(
            trade=trade,
            gives_value=30.0,
            receives_value=20.0,
            net_value=-10.0,
            gives_detail=[self._pv(1, 30.0)],
            receives_detail=[self._pv(10, 20.0)],
            recommendation="reject",
        )
        assert evaluation.trade is trade
        assert evaluation.gives_value == 30.0
        assert evaluation.receives_value == 20.0
        assert evaluation.net_value == -10.0
        assert len(evaluation.gives_detail) == 1
        assert len(evaluation.receives_detail) == 1
        assert evaluation.recommendation == "reject"


class TestCascadeRoster:
    def test_frozen(self) -> None:
        roster = CascadeRoster(picks=[], total_value=0.0)
        with pytest.raises(FrozenInstanceError):
            roster.total_value = 1.0  # type: ignore[misc]

    def test_fields(self) -> None:
        pick = DraftPick(
            round=1,
            pick=1,
            team_idx=0,
            player_id=1,
            player_name="Player A",
            position="OF",
            value=30.0,
        )
        roster = CascadeRoster(picks=[pick], total_value=30.0)
        assert len(roster.picks) == 1
        assert roster.picks[0].player_name == "Player A"
        assert roster.total_value == 30.0


class TestCascadeResult:
    def test_frozen(self) -> None:
        trade = PickTrade(gives=[1], receives=[4])
        before = CascadeRoster(picks=[], total_value=0.0)
        after = CascadeRoster(picks=[], total_value=0.0)
        result = CascadeResult(
            trade=trade,
            before=before,
            after=after,
            value_delta=0.0,
            recommendation="even",
        )
        with pytest.raises(FrozenInstanceError):
            result.value_delta = 5.0  # type: ignore[misc]

    def test_fields(self) -> None:
        trade = PickTrade(gives=[1], receives=[4])
        before = CascadeRoster(picks=[], total_value=50.0)
        after = CascadeRoster(picks=[], total_value=40.0)
        result = CascadeResult(
            trade=trade,
            before=before,
            after=after,
            value_delta=-10.0,
            recommendation="reject",
        )
        assert result.trade is trade
        assert result.before.total_value == 50.0
        assert result.after.total_value == 40.0
        assert result.value_delta == -10.0
        assert result.recommendation == "reject"
