import pytest

from fantasy_baseball_manager.domain.statcast_pitch import StatcastPitch


class TestStatcastPitch:
    def test_construct_with_required_fields(self) -> None:
        pitch = StatcastPitch(
            game_pk=718001,
            game_date="2024-06-15",
            batter_id=545361,
            pitcher_id=477132,
            at_bat_number=1,
            pitch_number=1,
        )
        assert pitch.game_pk == 718001
        assert pitch.game_date == "2024-06-15"
        assert pitch.batter_id == 545361
        assert pitch.pitcher_id == 477132
        assert pitch.at_bat_number == 1
        assert pitch.pitch_number == 1

    def test_optional_fields_default_to_none(self) -> None:
        pitch = StatcastPitch(
            game_pk=718001,
            game_date="2024-06-15",
            batter_id=545361,
            pitcher_id=477132,
            at_bat_number=1,
            pitch_number=1,
        )
        assert pitch.id is None
        assert pitch.pitch_type is None
        assert pitch.release_speed is None
        assert pitch.release_spin_rate is None
        assert pitch.pfx_x is None
        assert pitch.pfx_z is None
        assert pitch.plate_x is None
        assert pitch.plate_z is None
        assert pitch.zone is None
        assert pitch.events is None
        assert pitch.description is None
        assert pitch.launch_speed is None
        assert pitch.launch_angle is None
        assert pitch.hit_distance_sc is None
        assert pitch.barrel is None
        assert pitch.estimated_ba_using_speedangle is None
        assert pitch.estimated_woba_using_speedangle is None
        assert pitch.loaded_at is None

    def test_construct_with_all_fields(self) -> None:
        pitch = StatcastPitch(
            id=1,
            game_pk=718001,
            game_date="2024-06-15",
            batter_id=545361,
            pitcher_id=477132,
            at_bat_number=1,
            pitch_number=1,
            pitch_type="FF",
            release_speed=95.2,
            release_spin_rate=2400.0,
            pfx_x=-5.1,
            pfx_z=10.3,
            plate_x=0.5,
            plate_z=2.8,
            zone=5,
            events="single",
            description="hit_into_play",
            launch_speed=102.3,
            launch_angle=15.0,
            hit_distance_sc=250.0,
            barrel=1,
            estimated_ba_using_speedangle=0.620,
            estimated_woba_using_speedangle=0.850,
            loaded_at="2024-06-16T00:00:00+00:00",
        )
        assert pitch.id == 1
        assert pitch.pitch_type == "FF"
        assert pitch.release_speed == 95.2
        assert pitch.barrel == 1

    def test_frozen(self) -> None:
        pitch = StatcastPitch(
            game_pk=718001,
            game_date="2024-06-15",
            batter_id=545361,
            pitcher_id=477132,
            at_bat_number=1,
            pitch_number=1,
        )
        with pytest.raises(AttributeError):
            pitch.game_pk = 999  # type: ignore[misc]
