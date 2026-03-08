from dataclasses import dataclass


@dataclass(frozen=True)
class ReplacementProfile:
    position: str
    player_type: str
    stat_line: dict[str, float]
