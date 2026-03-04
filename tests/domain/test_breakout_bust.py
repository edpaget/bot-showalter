import pytest

from fantasy_baseball_manager.domain.breakout_bust import (
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
