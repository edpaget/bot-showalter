import pytest

from fantasy_baseball_manager.domain.pitching_stats import PitchingStats


class TestPitchingStats:
    def test_construct_with_required_fields(self) -> None:
        stats = PitchingStats(player_id=1, season=2024, source="fangraphs")
        assert stats.player_id == 1
        assert stats.season == 2024
        assert stats.source == "fangraphs"

    def test_optional_stat_fields_default_to_none(self) -> None:
        stats = PitchingStats(player_id=1, season=2024, source="fangraphs")
        assert stats.id is None
        assert stats.team_id is None
        assert stats.w is None
        assert stats.l is None
        assert stats.era is None
        assert stats.g is None
        assert stats.gs is None
        assert stats.sv is None
        assert stats.hld is None
        assert stats.ip is None
        assert stats.h is None
        assert stats.er is None
        assert stats.hr is None
        assert stats.bb is None
        assert stats.so is None
        assert stats.whip is None
        assert stats.k_per_9 is None
        assert stats.bb_per_9 is None
        assert stats.fip is None
        assert stats.xfip is None
        assert stats.war is None
        assert stats.loaded_at is None

    def test_construct_with_stats(self) -> None:
        stats = PitchingStats(
            player_id=1,
            season=2024,
            source="fangraphs",
            w=15,
            l=5,
            era=2.89,
            ip=200.1,
            so=250,
        )
        assert stats.w == 15
        assert stats.era == 2.89
        assert stats.so == 250

    def test_frozen(self) -> None:
        stats = PitchingStats(player_id=1, season=2024, source="fangraphs")
        with pytest.raises(AttributeError):
            stats.w = 10  # type: ignore[misc]
