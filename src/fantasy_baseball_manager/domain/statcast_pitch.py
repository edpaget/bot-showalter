from dataclasses import dataclass


@dataclass(frozen=True)
class StatcastPitch:
    game_pk: int
    game_date: str
    batter_id: int
    pitcher_id: int
    at_bat_number: int
    pitch_number: int
    id: int | None = None
    pitch_type: str | None = None
    release_speed: float | None = None
    release_spin_rate: float | None = None
    pfx_x: float | None = None
    pfx_z: float | None = None
    plate_x: float | None = None
    plate_z: float | None = None
    zone: int | None = None
    events: str | None = None
    description: str | None = None
    launch_speed: float | None = None
    launch_angle: float | None = None
    hit_distance_sc: float | None = None
    barrel: int | None = None
    estimated_ba_using_speedangle: float | None = None
    estimated_woba_using_speedangle: float | None = None
    estimated_slg_using_speedangle: float | None = None
    loaded_at: str | None = None
