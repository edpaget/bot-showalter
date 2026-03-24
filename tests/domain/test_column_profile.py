import pytest

from fantasy_baseball_manager.domain.column_profile import ColumnProfile
from fantasy_baseball_manager.domain.identity import PlayerType


class TestColumnProfile:
    def test_construct_with_all_fields(self) -> None:
        profile = ColumnProfile(
            column="launch_speed",
            season=2023,
            player_type=PlayerType.BATTER,
            count=100,
            null_count=5,
            null_pct=4.762,
            mean=88.5,
            median=89.0,
            std=5.2,
            min=70.0,
            max=105.0,
            p10=81.0,
            p25=85.0,
            p75=93.0,
            p90=96.0,
            skewness=-0.3,
        )
        assert profile.column == "launch_speed"
        assert profile.season == 2023
        assert profile.player_type == "batter"
        assert profile.count == 100
        assert profile.null_count == 5
        assert profile.null_pct == 4.762
        assert profile.mean == 88.5
        assert profile.median == 89.0
        assert profile.std == 5.2
        assert profile.min == 70.0
        assert profile.max == 105.0
        assert profile.p10 == 81.0
        assert profile.p25 == 85.0
        assert profile.p75 == 93.0
        assert profile.p90 == 96.0
        assert profile.skewness == -0.3

    def test_frozen(self) -> None:
        profile = ColumnProfile(
            column="launch_speed",
            season=2023,
            player_type=PlayerType.BATTER,
            count=100,
            null_count=5,
            null_pct=4.762,
            mean=88.5,
            median=89.0,
            std=5.2,
            min=70.0,
            max=105.0,
            p10=81.0,
            p25=85.0,
            p75=93.0,
            p90=96.0,
            skewness=-0.3,
        )
        with pytest.raises(AttributeError):
            profile.column = "barrel"  # type: ignore[misc]

    def test_pitcher_player_type(self) -> None:
        profile = ColumnProfile(
            column="release_speed",
            season=2024,
            player_type=PlayerType.PITCHER,
            count=200,
            null_count=0,
            null_pct=0.0,
            mean=93.1,
            median=93.5,
            std=2.8,
            min=85.0,
            max=101.0,
            p10=89.5,
            p25=91.0,
            p75=95.0,
            p90=97.0,
            skewness=-0.1,
        )
        assert profile.player_type == "pitcher"
        assert profile.null_count == 0
        assert profile.null_pct == 0.0
