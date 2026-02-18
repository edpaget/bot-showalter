from dataclasses import dataclass


@dataclass(frozen=True)
class SprintSpeed:
    mlbam_id: int
    season: int
    sprint_speed: float | None = None
    hp_to_1b: float | None = None
    bolts: int | None = None
    competitive_runs: int | None = None
    id: int | None = None
    loaded_at: str | None = None
