from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerSummary:
    player_id: int
    name: str
    team: str
    age: int | None
    primary_position: str
    bats: str | None
    throws: str | None
    experience: int
