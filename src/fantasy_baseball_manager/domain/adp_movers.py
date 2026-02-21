from dataclasses import dataclass


@dataclass(frozen=True)
class ADPMover:
    player_name: str
    position: str
    current_rank: int
    previous_rank: int
    rank_delta: int
    direction: str


@dataclass(frozen=True)
class ADPMoversReport:
    season: int
    provider: str
    current_as_of: str
    previous_as_of: str
    risers: list[ADPMover]
    fallers: list[ADPMover]
    new_entries: list[ADPMover]
    dropped_entries: list[ADPMover]
