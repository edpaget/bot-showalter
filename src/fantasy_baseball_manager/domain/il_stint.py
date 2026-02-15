from dataclasses import dataclass


@dataclass(frozen=True)
class ILStint:
    player_id: int
    season: int
    start_date: str
    il_type: str
    id: int | None = None
    end_date: str | None = None
    days: int | None = None
    injury_location: str | None = None
    transaction_type: str | None = None
    loaded_at: str | None = None
