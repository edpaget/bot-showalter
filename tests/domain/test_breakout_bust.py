import pytest

from fantasy_baseball_manager.domain.breakout_bust import (
    BreakoutPrediction,
    LabelConfig,
    LabeledSeason,
    OutcomeLabel,
)


class TestOutcomeLabel:
    def test_values(self) -> None:
        assert OutcomeLabel.BREAKOUT == "breakout"
        assert OutcomeLabel.BUST == "bust"
        assert OutcomeLabel.NEUTRAL == "neutral"

    def test_is_str(self) -> None:
        assert isinstance(OutcomeLabel.BREAKOUT, str)

    def test_from_string(self) -> None:
        assert OutcomeLabel("breakout") is OutcomeLabel.BREAKOUT


class TestLabelConfig:
    def test_defaults(self) -> None:
        config = LabelConfig()
        assert config.breakout_threshold == 30
        assert config.bust_threshold == -30
        assert config.min_adp_rank == 300

    def test_custom_values(self) -> None:
        config = LabelConfig(breakout_threshold=50, bust_threshold=-50, min_adp_rank=200)
        assert config.breakout_threshold == 50
        assert config.bust_threshold == -50
        assert config.min_adp_rank == 200

    def test_frozen(self) -> None:
        config = LabelConfig()
        with pytest.raises(AttributeError):
            config.breakout_threshold = 99  # type: ignore[misc]


class TestLabeledSeason:
    def test_fields(self) -> None:
        ls = LabeledSeason(
            player_id=1,
            season=2023,
            player_type="batter",
            adp_rank=50,
            adp_pick=50.0,
            actual_value_rank=20,
            rank_delta=30,
            label=OutcomeLabel.BREAKOUT,
        )
        assert ls.player_id == 1
        assert ls.season == 2023
        assert ls.player_type == "batter"
        assert ls.adp_rank == 50
        assert ls.adp_pick == 50.0
        assert ls.actual_value_rank == 20
        assert ls.rank_delta == 30
        assert ls.label is OutcomeLabel.BREAKOUT

    def test_frozen(self) -> None:
        ls = LabeledSeason(
            player_id=1,
            season=2023,
            player_type="batter",
            adp_rank=50,
            adp_pick=50.0,
            actual_value_rank=20,
            rank_delta=30,
            label=OutcomeLabel.BREAKOUT,
        )
        with pytest.raises(AttributeError):
            ls.label = OutcomeLabel.BUST  # type: ignore[misc]


class TestBreakoutPrediction:
    def test_fields(self) -> None:
        pred = BreakoutPrediction(
            player_id=42,
            player_name="Mike Trout",
            player_type="batter",
            position="OF",
            p_breakout=0.6,
            p_bust=0.1,
            p_neutral=0.3,
            top_features=[("age", 0.25), ("avg_exit_velo", 0.15)],
        )
        assert pred.player_id == 42
        assert pred.player_name == "Mike Trout"
        assert pred.player_type == "batter"
        assert pred.position == "OF"
        assert pred.p_breakout == 0.6
        assert pred.p_bust == 0.1
        assert pred.p_neutral == 0.3
        assert pred.top_features == [("age", 0.25), ("avg_exit_velo", 0.15)]

    def test_frozen(self) -> None:
        pred = BreakoutPrediction(
            player_id=42,
            player_name="Mike Trout",
            player_type="batter",
            position="OF",
            p_breakout=0.6,
            p_bust=0.1,
            p_neutral=0.3,
            top_features=[],
        )
        with pytest.raises(AttributeError):
            pred.p_breakout = 0.9  # type: ignore[misc]
