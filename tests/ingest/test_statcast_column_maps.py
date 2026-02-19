from typing import Any

from fantasy_baseball_manager.ingest.column_maps import statcast_pitch_mapper


def _make_row(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "game_pk": 718001,
        "game_date": "2024-06-15",
        "batter": 545361,
        "pitcher": 477132,
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitch_type": "FF",
        "release_speed": 95.2,
        "release_spin_rate": 2400.0,
        "pfx_x": -5.1,
        "pfx_z": 10.3,
        "plate_x": 0.5,
        "plate_z": 2.8,
        "zone": 5,
        "events": "single",
        "description": "hit_into_play",
        "launch_speed": 102.3,
        "launch_angle": 15.0,
        "hit_distance_sc": 250.0,
        "barrel": 1,
        "estimated_ba_using_speedangle": 0.620,
        "estimated_woba_using_speedangle": 0.850,
        "estimated_slg_using_speedangle": 0.750,
        "hc_x": 105.3,
        "hc_y": 160.2,
        "stand": "R",
        "release_extension": 6.3,
    }
    return {**defaults, **overrides}


class TestStatcastPitchMapper:
    def test_complete_row(self) -> None:
        result = statcast_pitch_mapper(_make_row())
        assert result is not None
        assert result.game_pk == 718001
        assert result.game_date == "2024-06-15"
        assert result.batter_id == 545361
        assert result.pitcher_id == 477132
        assert result.at_bat_number == 1
        assert result.pitch_number == 1
        assert result.pitch_type == "FF"
        assert result.release_speed == 95.2
        assert result.release_spin_rate == 2400.0
        assert result.pfx_x == -5.1
        assert result.pfx_z == 10.3
        assert result.plate_x == 0.5
        assert result.plate_z == 2.8
        assert result.zone == 5
        assert result.events == "single"
        assert result.description == "hit_into_play"
        assert result.launch_speed == 102.3
        assert result.launch_angle == 15.0
        assert result.hit_distance_sc == 250.0
        assert result.barrel == 1
        assert result.estimated_ba_using_speedangle == 0.620
        assert result.estimated_woba_using_speedangle == 0.850
        assert result.estimated_slg_using_speedangle == 0.750
        assert result.hc_x == 105.3
        assert result.hc_y == 160.2
        assert result.stand == "R"
        assert result.release_extension == 6.3

    def test_nan_optional_fields_become_none(self) -> None:
        result = statcast_pitch_mapper(
            _make_row(
                pitch_type=float("nan"),
                release_speed=float("nan"),
                launch_speed=float("nan"),
                barrel=float("nan"),
                events=float("nan"),
            )
        )
        assert result is not None
        assert result.pitch_type is None
        assert result.release_speed is None
        assert result.launch_speed is None
        assert result.barrel is None
        assert result.events is None

    def test_nan_batter_returns_none(self) -> None:
        result = statcast_pitch_mapper(_make_row(batter=float("nan")))
        assert result is None

    def test_nan_pitcher_returns_none(self) -> None:
        result = statcast_pitch_mapper(_make_row(pitcher=float("nan")))
        assert result is None

    def test_nan_at_bat_number_returns_none(self) -> None:
        result = statcast_pitch_mapper(_make_row(at_bat_number=float("nan")))
        assert result is None

    def test_nan_pitch_number_returns_none(self) -> None:
        result = statcast_pitch_mapper(_make_row(pitch_number=float("nan")))
        assert result is None

    def test_nan_game_pk_returns_none(self) -> None:
        result = statcast_pitch_mapper(_make_row(game_pk=float("nan")))
        assert result is None

    def test_barrel_as_int(self) -> None:
        result = statcast_pitch_mapper(_make_row(barrel=1.0))
        assert result is not None
        assert result.barrel == 1
        assert isinstance(result.barrel, int)

    def test_string_fields(self) -> None:
        result = statcast_pitch_mapper(_make_row(events="home_run", description="hit_into_play"))
        assert result is not None
        assert result.events == "home_run"
        assert result.description == "hit_into_play"

    def test_missing_optional_columns(self) -> None:
        row: dict[str, Any] = {
            "game_pk": 718001,
            "game_date": "2024-06-15",
            "batter": 545361,
            "pitcher": 477132,
            "at_bat_number": 1,
            "pitch_number": 1,
        }
        result = statcast_pitch_mapper(row)
        assert result is not None
        assert result.pitch_type is None
        assert result.release_speed is None
        assert result.barrel is None

    def test_string_values_from_csv(self) -> None:
        row = _make_row(
            game_pk="718001",
            batter="545361",
            pitcher="477132",
            at_bat_number="1",
            pitch_number="1",
            release_speed="95.2",
            release_spin_rate="2400.0",
            pfx_x="-5.1",
            pfx_z="10.3",
            plate_x="0.5",
            plate_z="2.8",
            zone="5",
            launch_speed="102.3",
            launch_angle="15.0",
            hit_distance_sc="250.0",
            barrel="1",
            estimated_ba_using_speedangle="0.620",
            estimated_woba_using_speedangle="0.850",
            estimated_slg_using_speedangle="0.750",
            hc_x="105.3",
            hc_y="160.2",
            release_extension="6.3",
        )
        result = statcast_pitch_mapper(row)
        assert result is not None
        assert result.game_pk == 718001
        assert result.batter_id == 545361
        assert result.pitcher_id == 477132
        assert result.release_speed == 95.2
        assert result.zone == 5
        assert result.barrel == 1

    def test_none_for_missing_optional_fields(self) -> None:
        row = _make_row(
            pitch_type=None,
            release_speed=None,
            release_spin_rate=None,
            launch_speed=None,
            launch_angle=None,
            barrel=None,
            events=None,
            description=None,
            hc_x=None,
            hc_y=None,
            stand=None,
            release_extension=None,
        )
        result = statcast_pitch_mapper(row)
        assert result is not None
        assert result.pitch_type is None
        assert result.release_speed is None
        assert result.release_spin_rate is None
        assert result.launch_speed is None
        assert result.barrel is None
        assert result.events is None
        assert result.stand is None
