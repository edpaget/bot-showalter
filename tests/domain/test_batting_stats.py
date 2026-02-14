import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats


class TestBattingStats:
    def test_construct_with_required_fields(self) -> None:
        stats = BattingStats(player_id=1, season=2024, source="fangraphs")
        assert stats.player_id == 1
        assert stats.season == 2024
        assert stats.source == "fangraphs"

    def test_optional_stat_fields_default_to_none(self) -> None:
        stats = BattingStats(player_id=1, season=2024, source="fangraphs")
        assert stats.id is None
        assert stats.team_id is None
        assert stats.pa is None
        assert stats.ab is None
        assert stats.h is None
        assert stats.doubles is None
        assert stats.triples is None
        assert stats.hr is None
        assert stats.rbi is None
        assert stats.r is None
        assert stats.sb is None
        assert stats.cs is None
        assert stats.bb is None
        assert stats.so is None
        assert stats.hbp is None
        assert stats.sf is None
        assert stats.sh is None
        assert stats.gdp is None
        assert stats.ibb is None
        assert stats.avg is None
        assert stats.obp is None
        assert stats.slg is None
        assert stats.ops is None
        assert stats.woba is None
        assert stats.wrc_plus is None
        assert stats.war is None
        assert stats.loaded_at is None

    def test_construct_with_stats(self) -> None:
        stats = BattingStats(
            player_id=1,
            season=2024,
            source="fangraphs",
            pa=600,
            ab=520,
            h=160,
            hr=35,
            avg=0.308,
            ops=0.950,
        )
        assert stats.pa == 600
        assert stats.hr == 35
        assert stats.avg == 0.308

    def test_frozen(self) -> None:
        stats = BattingStats(player_id=1, season=2024, source="fangraphs")
        with pytest.raises(AttributeError):
            stats.pa = 500  # type: ignore[misc]
