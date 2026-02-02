import pytest

from fantasy_baseball_manager.draft.models import RosterConfig
from fantasy_baseball_manager.draft.simulation_models import (
    DraftRule,
    DraftStrategy,
    SimulationConfig,
    SimulationPick,
    SimulationResult,
    TeamConfig,
    TeamResult,
)
from fantasy_baseball_manager.valuation.models import StatCategory


def _make_strategy(
    name: str = "test",
    weights: dict[StatCategory, float] | None = None,
    rules: tuple[DraftRule, ...] = (),
    noise_scale: float = 0.15,
) -> DraftStrategy:
    return DraftStrategy(
        name=name,
        category_weights=weights or {},
        rules=rules,
        noise_scale=noise_scale,
    )


class TestDraftStrategy:
    def test_default_noise_scale(self) -> None:
        s = _make_strategy()
        assert s.noise_scale == 0.15

    def test_custom_noise_scale(self) -> None:
        s = _make_strategy(noise_scale=0.3)
        assert s.noise_scale == 0.3

    def test_negative_noise_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="noise_scale"):
            _make_strategy(noise_scale=-0.1)

    def test_empty_weights_allowed(self) -> None:
        s = _make_strategy(weights={})
        assert s.category_weights == {}

    def test_frozen(self) -> None:
        s = _make_strategy()
        with pytest.raises(AttributeError):
            s.name = "changed"  # type: ignore[misc]


class TestTeamConfig:
    def test_default_keepers_empty(self) -> None:
        tc = TeamConfig(team_id=1, name="Team 1", strategy=_make_strategy())
        assert tc.keepers == ()

    def test_keepers_set(self) -> None:
        tc = TeamConfig(team_id=1, name="Team 1", strategy=_make_strategy(), keepers=("p1", "p2"))
        assert tc.keepers == ("p1", "p2")


class TestSimulationConfig:
    def test_seed_defaults_to_none(self) -> None:
        cfg = SimulationConfig(
            teams=(),
            roster_config=RosterConfig(slots=()),
            total_rounds=20,
        )
        assert cfg.seed is None

    def test_seed_set(self) -> None:
        cfg = SimulationConfig(
            teams=(),
            roster_config=RosterConfig(slots=()),
            total_rounds=20,
            seed=42,
        )
        assert cfg.seed == 42


class TestSimulationPick:
    def test_fields(self) -> None:
        pick = SimulationPick(
            overall_pick=1,
            round_number=1,
            pick_in_round=1,
            team_id=1,
            team_name="Team 1",
            player_id="p1",
            player_name="Player 1",
            position="1B",
            adjusted_value=5.0,
        )
        assert pick.overall_pick == 1
        assert pick.player_id == "p1"
        assert pick.position == "1B"

    def test_position_can_be_none(self) -> None:
        pick = SimulationPick(
            overall_pick=1,
            round_number=1,
            pick_in_round=1,
            team_id=1,
            team_name="Team 1",
            player_id="p1",
            player_name="Player 1",
            position=None,
            adjusted_value=5.0,
        )
        assert pick.position is None


class TestTeamResult:
    def test_default_category_totals_empty(self) -> None:
        tr = TeamResult(
            team_id=1,
            team_name="Team 1",
            strategy_name="test",
            picks=(),
        )
        assert tr.category_totals == {}


class TestSimulationResult:
    def test_fields(self) -> None:
        cfg = SimulationConfig(
            teams=(),
            roster_config=RosterConfig(slots=()),
            total_rounds=20,
        )
        result = SimulationResult(
            pick_log=(),
            team_results=(),
            config=cfg,
        )
        assert result.pick_log == ()
        assert result.team_results == ()
        assert result.config is cfg
