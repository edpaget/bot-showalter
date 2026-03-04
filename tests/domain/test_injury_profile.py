from fantasy_baseball_manager.domain import ILStint, InjuryProfile


class TestInjuryProfile:
    def test_construction_defaults(self) -> None:
        profile = InjuryProfile(
            player_id=1,
            seasons_tracked=5,
            total_stints=0,
            total_days_lost=0,
            avg_days_per_season=0.0,
            max_days_in_season=0,
            pct_seasons_with_il=0.0,
        )
        assert profile.player_id == 1
        assert profile.seasons_tracked == 5
        assert profile.total_stints == 0
        assert profile.injury_locations == {}
        assert profile.recent_stints == []

    def test_construction_with_all_fields(self) -> None:
        stint = ILStint(player_id=1, season=2024, start_date="2024-05-01", il_type="10-day")
        profile = InjuryProfile(
            player_id=1,
            seasons_tracked=3,
            total_stints=2,
            total_days_lost=40,
            avg_days_per_season=13.3,
            max_days_in_season=30,
            pct_seasons_with_il=0.667,
            injury_locations={"shoulder": 1, "hamstring": 1},
            recent_stints=[stint],
        )
        assert profile.total_days_lost == 40
        assert profile.injury_locations == {"shoulder": 1, "hamstring": 1}
        assert len(profile.recent_stints) == 1

    def test_frozen(self) -> None:
        profile = InjuryProfile(
            player_id=1,
            seasons_tracked=1,
            total_stints=0,
            total_days_lost=0,
            avg_days_per_season=0.0,
            max_days_in_season=0,
            pct_seasons_with_il=0.0,
        )
        try:
            profile.total_stints = 5  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass
